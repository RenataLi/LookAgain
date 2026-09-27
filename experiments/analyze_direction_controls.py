"""Paired image-level analysis of the complete 17-action direction-control run.

No model loading or answer rescoring. Expected condition accuracy averages all
four recorded quadrant outcomes within each image, then averages images. Four
presentations of an image are never treated as four independent observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import random
import re
import statistics

from PIL import Image


CONDITIONS = ("named", "neutral", "sham", "frame")
REGIONS = ("tl", "tr", "bl", "br")
LOCATIONS = {"tl": "upper-left", "tr": "upper-right", "bl": "lower-left", "br": "lower-right"}
BOXES = {"tl": [0.0, 0.0, 0.6, 0.6], "tr": [0.4, 0.0, 1.0, 0.6],
         "bl": [0.0, 0.4, 0.6, 1.0], "br": [0.4, 0.4, 1.0, 1.0]}
ACTIONS = ("direct", *(f"{condition}_{region}" for condition in CONDITIONS for region in REGIONS))
IDENTITY_KEYS = ("config", "manifest_sha256", "code_sha256", "model", "runtime", "example_ids")
PRIMARY_CONTRASTS = (("neutral", "named"), ("named", "sham"), ("frame", "named"))
DIRECTION_PATTERN = re.compile(r"\b(left|right|leftmost|rightmost)\b")
BOOTSTRAP_SAMPLES = 10000
SEED = 20260925
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CAVEATS = [
    "The sample is the existing development convenience panel, not a new held-out dataset. "
    "No controller, transfer claim or dataset-wide representative estimate is established.",
    "Expected condition accuracy means a uniformly sampled quadrant, averaged analytically over "
    "four observed outcomes per image. This is not an oracle or a best-quadrant selector.",
    "For each condition there are 4N presentations but only N independent sampling units. "
    "The bootstrap resamples images and retains all 17 paired action outcomes together.",
    "Primary contrasts are neutral minus named, named minus sham, and frame minus named. "
    "Reported 95% percentile intervals are unadjusted for multiple comparisons and describe "
    "conditional image-sampling uncertainty, not generation or hardware variability. No p-values are computed.",
    "The four conditions are not a full factorial design: a neutral sham condition is absent. "
    "Named minus sham includes effects of misleading image-text incongruity, and is not proof "
    "of causal word copying or of new visual evidence.",
    "The question-text subgroup is fixed by the declared lexical regex, without answer labels. "
    "Target-answer left/right groups are secondary label-conditioned descriptions, never inputs to inference.",
    "Direction-switch and cue-alignment measures use conservative normalization of recorded parsed answers. "
    "They do not rescore correctness. Alignment is descriptive and can reflect the distribution of targets and scenes.",
    "All original correctness labels are retained. Invalidly parsed answers are included as incorrect; "
    "failed, missing, duplicate or undeclared action records prevent analysis of the run.",
    "Timing is descriptive for each fixed condition/quadrant. Follow-up full-path time is direct plus "
    "incremental follow-up time; sequential peak memory is their maximum. No pooled deployment or routing efficiency claim is made.",
    "A v1-exclusive subset, when supplied, was selected using previous outcomes on these images. "
    "It is explicitly exploratory and has no inferential intervals or independent-confirmation claim.",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{number} must be a JSON object")
            rows.append(row)
    return rows


def unique_strings(values, description):
    if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
        raise ValueError(f"{description} must be a nonempty list of nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{description} must be unique")


def normalize(text):
    # Same conservative rule as v1; this is descriptive, never correctness grading.
    return " ".join(text.strip().casefold().rstrip(".!?").split()) if isinstance(text, str) else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def percentile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def describe_numbers(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p95": percentile(values, 0.95), "min": min(values), "max": max(values)} if values else None


def validate_run(run: Path, manifest: Path, allow_smoke=False):
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or any(key not in metadata for key in IDENTITY_KEYS):
        raise ValueError("Run identity is missing required fields")
    identity = {key: metadata[key] for key in IDENTITY_KEYS}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("Run fingerprint does not match its six-field identity")
    if not isinstance(metadata["code_sha256"], str) or not HEX64.fullmatch(metadata["code_sha256"]):
        raise ValueError("Run source-code hash is invalid")
    if sha256_file(manifest) != metadata["manifest_sha256"]:
        raise ValueError("Manifest hash differs from the run identity")
    planned = metadata["example_ids"]
    unique_strings(planned, "Planned example IDs")
    minimum = metadata["config"].get("analysis", {}).get("minimum_complete_examples_for_primary_report", 1)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("Invalid minimum complete primary cohort size")
    if len(planned) < minimum and not allow_smoke:
        raise ValueError("Planned run is smaller than the protocol's minimum complete primary cohort")
    analysis_config = metadata["config"].get("analysis", {})
    for field, expected_value in (("bootstrap_samples", BOOTSTRAP_SAMPLES), ("bootstrap_seed", SEED),
                                  ("question_direction_regex", DIRECTION_PATTERN.pattern)):
        if field in analysis_config and analysis_config[field] != expected_value:
            raise ValueError(f"Frozen analysis contract differs on {field}")
    actions = metadata["config"].get("actions")
    if not isinstance(actions, list) or len(actions) != len(ACTIONS) or set(actions) != set(ACTIONS):
        raise ValueError("Run must declare all 17 direction-control actions exactly once")
    if metadata["config"].get("model_id") != metadata["model"].get("model_id"):
        raise ValueError("Config and model identity disagree")
    token_limit = metadata["config"].get("answer_max_tokens")
    if isinstance(token_limit, bool) or not isinstance(token_limit, int) or token_limit < 1:
        raise ValueError("Config answer_max_tokens must be a positive integer")
    manifest_rows = read_jsonl(manifest)
    unique_strings([row.get("example_id") for row in manifest_rows], "Manifest example IDs")
    unique_strings([row.get("image_id") for row in manifest_rows], "Manifest image IDs")
    sources = {row["example_id"]: row for row in manifest_rows}
    if set(planned) - set(sources):
        raise ValueError("Planned examples are absent from the matching manifest")

    records = read_jsonl(run / "records.jsonl")
    expected = {(example_id, action) for example_id in planned for action in ACTIONS}
    grouped = defaultdict(dict)
    seen = set()
    for index, row in enumerate(records):
        for field in ("example_id", "image_id", "action"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Record {index}: invalid {field}")
        key = (row["example_id"], row["action"])
        if key not in expected:
            raise ValueError(f"Undeclared example/action record: {key}")
        if key in seen:
            raise ValueError(f"Duplicate action record: {key}")
        seen.add(key)
        if row.get("status") != "ok" or not isinstance(row.get("correct"), bool):
            raise ValueError(f"Failed or invalid correctness record: {key}")
        if not isinstance(row.get("parse_valid"), bool):
            raise ValueError(f"Missing parse validity: {key}")
        if not isinstance(row.get("generation_truncated"), bool):
            raise ValueError(f"Missing or invalid generation_truncated flag: {key}")
        generated_tokens = row.get("generated_tokens")
        if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int) or not 1 <= generated_tokens <= token_limit:
            raise ValueError(f"Invalid generated_tokens count for configured limit {token_limit}: {key}")
        if not row["parse_valid"] and row["correct"]:
            raise ValueError(f"Invalidly parsed answer cannot be marked correct: {key}")
        if row["parse_valid"] and (not isinstance(row.get("predicted_answer"), str) or not row["predicted_answer"].strip()):
            raise ValueError(f"Parsed answer must be nonempty: {key}")
        for field in ("elapsed_s", "peak_memory_gib"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {field}: {key}")
        if row["action"] == "direct":
            condition, region = "direct", None
        else:
            condition, region = row["action"].split("_")
        if row.get("condition") != condition or row.get("region") != region:
            raise ValueError(f"Condition/region differs from action identity: {key}")
        source = sources[row["example_id"]]
        for record_field, source_field in (("image_id", "image_id"), ("question", "question"), ("target_answer", "answer")):
            if row.get(record_field) != source.get(source_field):
                raise ValueError(f"Record and manifest disagree on {record_field}: {key}")
        grouped[row["example_id"]][row["action"]] = row
    if seen != expected:
        raise ValueError(f"Incomplete planned run: {len(expected - seen)} missing example/action records")
    # Audit the intervention before interpreting any prompt/pixel comparisons.
    prompts = metadata["config"].get("condition_prompts", {})
    if set(prompts) != set(CONDITIONS) or prompts["named"] != prompts["sham"] or "{location}" in prompts["neutral"]:
        raise ValueError("Condition prompt contract is inconsistent")
    for example_id in planned:
        group = grouped[example_id]
        direct_hashes = group["direct"].get("image_rgb_sha256")
        if not isinstance(direct_hashes, list) or len(direct_hashes) != 1 or not isinstance(direct_hashes[0], str) or not HEX64.fullmatch(direct_hashes[0]):
            raise ValueError(f"Invalid direct pixel hash: {example_id}")
        overview_hash = direct_hashes[0]
        for action in ACTIONS[1:]:
            row = group[action]
            condition, region = action.split("_")
            hashes = row.get("image_rgb_sha256")
            if not isinstance(hashes, list) or len(hashes) != 2 or any(not isinstance(x, str) or not HEX64.fullmatch(x) for x in hashes):
                raise ValueError(f"Invalid follow-up pixel hashes: {example_id}/{action}")
            if hashes[0] != overview_hash or row.get("overview_size") != group["direct"].get("overview_size"):
                raise ValueError(f"Overview pixel integrity mismatch: {example_id}/{action}")
            if row.get("followup_prompt") != prompts[condition].format(location=LOCATIONS[region]):
                raise ValueError(f"Realized follow-up prompt differs from config: {example_id}/{action}")
            if row.get("assigned_region_box") != BOXES[region] or row.get("claimed_crop_box") != (None if condition == "neutral" else BOXES[region]):
                raise ValueError(f"Claimed/assigned crop identity mismatch: {example_id}/{action}")
            if condition == "sham":
                if (hashes[1] != overview_hash or row.get("actual_second_view_box") != [0.0, 0.0, 1.0, 1.0]
                        or row.get("additional_view_kind") != "exact_overview_repeat"
                        or row.get("additional_size") != group["direct"].get("overview_size")):
                    raise ValueError(f"Sham is not an exact overview repeat: {example_id}/{action}")
            elif row.get("actual_second_view_box") != BOXES[region] or row.get("additional_view_kind") != "true_crop":
                raise ValueError(f"Actual crop identity mismatch: {example_id}/{action}")
        for region in REGIONS:
            rows = [group[f"{condition}_{region}"] for condition in ("named", "neutral", "frame")]
            if len({row["image_rgb_sha256"][1] for row in rows}) != 1 or len({tuple(row.get("additional_size", [])) for row in rows}) != 1:
                raise ValueError(f"True-crop pixels differ across conditions: {example_id}/{region}")
    image_ids = [grouped[key]["direct"]["image_id"] for key in planned]
    unique_strings(image_ids, "Completed image IDs")
    byte_hashes, pixel_hashes = set(), set()
    for example_id in planned:
        source = sources[example_id]
        relative = source.get("image_path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or PureWindowsPath(relative).drive:
            raise ValueError("Image path must be relative to the manifest")
        path = (manifest.parent / relative).resolve()
        if not path.is_relative_to(manifest.parent):
            raise ValueError("Image path escapes the manifest directory")
        digest = sha256_file(path)
        if digest != source.get("image_sha256"):
            raise ValueError(f"Source image checksum differs: {example_id}")
        with Image.open(path) as image:
            image.load()
            pixels = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode() + image.convert("RGB").tobytes()).hexdigest()
        if digest in byte_hashes or pixels in pixel_hashes:
            raise ValueError("Duplicate encoded or decoded source images in the planned cohort")
        byte_hashes.add(digest)
        pixel_hashes.add(pixels)
    return metadata, sources, [grouped[key] for key in planned], len(manifest_rows)


def transition_metrics(groups, actions):
    """Presentation counts with the independent image denominator kept explicit."""
    cross = Counter((group["direct"]["correct"], group[action]["correct"]) for group in groups for action in actions)
    n_images, per_image = len(groups), len(actions)
    total = n_images * per_image
    fixes, harms = cross[(False, True)], cross[(True, False)]
    baseline_correct = sum(group["direct"]["correct"] for group in groups)
    correct = fixes + cross[(True, True)]
    return {
        "n_images": n_images, "n_presentations": total, "presentations_per_image": per_image,
        "correct_presentations": correct, "expected_accuracy": ratio(correct, total),
        "direct_correct_images": baseline_correct, "direct_accuracy": ratio(baseline_correct, n_images),
        "delta_vs_direct_pp": 100 * (correct / total - baseline_correct / n_images) if total else None,
        "correct_to_correct_presentations": cross[(True, True)], "wrong_to_wrong_presentations": cross[(False, False)],
        "fix_presentations": fixes, "harm_presentations": harms,
        "fix_fraction_all_presentations": ratio(fixes, total), "harm_fraction_all_presentations": ratio(harms, total),
        "fix_rate_given_direct_wrong": ratio(fixes, (n_images - baseline_correct) * per_image),
        "fix_rate_denominator_presentations": (n_images - baseline_correct) * per_image,
        "harm_rate_given_direct_correct": ratio(harms, baseline_correct * per_image),
        "harm_rate_denominator_presentations": baseline_correct * per_image,
        "unique_images_with_any_fix": sum(not group["direct"]["correct"] and any(group[a]["correct"] for a in actions) for group in groups),
        "unique_images_with_any_harm": sum(group["direct"]["correct"] and any(not group[a]["correct"] for a in actions) for group in groups),
        "invalid_parse_presentations": sum(not group[a]["parse_valid"] for group in groups for a in actions),
    }


def primary_contrasts(groups, bootstrap=True):
    n = len(groups)
    vectors = {condition: [sum(group[f"{condition}_{region}"]["correct"] for region in REGIONS) / 4 for group in groups]
               for condition in CONDITIONS}
    differences = {f"{left}_minus_{right}": [a - b for a, b in zip(vectors[left], vectors[right])]
                   for left, right in PRIMARY_CONTRASTS}
    result = {}
    for left, right in PRIMARY_CONTRASTS:
        key = f"{left}_minus_{right}"
        values = differences[key]
        result[key] = {"left": left, "reference": right, "n_images": n,
                       "delta_expected_accuracy_pp": 100 * statistics.mean(values) if n else None,
                       "delta_ci_95_pp": None,
                       "images_with_positive_difference": sum(value > 0 for value in values),
                       "images_with_negative_difference": sum(value < 0 for value in values),
                       "images_with_zero_difference": sum(value == 0 for value in values),
                       "bootstrap_samples": BOOTSTRAP_SAMPLES if bootstrap and n >= 2 else 0}
    if bootstrap and n >= 2:
        rng = random.Random(SEED)
        draws = {key: [] for key in differences}
        for _ in range(BOOTSTRAP_SAMPLES):
            multiplicities = Counter(rng.randrange(n) for _ in range(n))
            for key, values in differences.items():
                draws[key].append(100 * sum(values[index] * count for index, count in multiplicities.items()) / n)
        for key in differences:
            result[key]["delta_ci_95_pp"] = [percentile(draws[key], p) for p in (0.025, 0.975)]
    return result


def predicted_direction(row):
    value = normalize(row.get("predicted_answer")) if row.get("parse_valid") else None
    return value if value in ("left", "right") else None


def direction_metrics(groups, condition, regions=REGIONS):
    switches = Counter()
    directional_pairs = 0
    directional_answers = aligned = 0
    target_matches = target_directional_answers = 0
    left_answers = right_answers = 0
    target_direction_pairs = Counter()
    for group in groups:
        initial = predicted_direction(group["direct"])
        target = normalize(group["direct"]["target_answer"])
        for region in regions:
            value = predicted_direction(group[f"{condition}_{region}"])
            side = "left" if region in ("tl", "bl") else "right"
            if initial and value:
                directional_pairs += 1
                switches[(initial, value)] += 1
            if value:
                directional_answers += 1
                left_answers += value == "left"
                right_answers += value == "right"
                aligned += value == side
                if target in ("left", "right"):
                    target_directional_answers += 1
                    target_matches += value == target
                    target_direction_pairs[(target, value)] += 1
    presentations = len(groups) * len(regions)
    has_text_cue = condition in ("named", "sham", "frame")
    return {
        "n_images": len(groups), "n_presentations": presentations,
        "both_direct_and_followup_are_directional_count": directional_pairs,
        "direct_left_to_followup_right": switches[("left", "right")],
        "direct_right_to_followup_left": switches[("right", "left")],
        "direct_left_to_followup_left": switches[("left", "left")],
        "direct_right_to_followup_right": switches[("right", "right")],
        "left_to_right_rate_among_direct_left_directional_pairs": ratio(switches[("left", "right")], switches[("left", "left")] + switches[("left", "right")]),
        "left_to_right_denominator": switches[("left", "left")] + switches[("left", "right")],
        "right_to_left_rate_among_direct_right_directional_pairs": ratio(switches[("right", "left")], switches[("right", "right")] + switches[("right", "left")]),
        "right_to_left_denominator": switches[("right", "right")] + switches[("right", "left")],
        "left_answers": left_answers, "right_answers": right_answers,
        "directional_answer_count": directional_answers,
        "nondirectional_or_invalid_answer_count": presentations - directional_answers,
        "region_side_alignment_count": aligned,
        "region_side_alignment_rate_among_directional_answers": ratio(aligned, directional_answers),
        "region_side_alignment_rate_all_presentations": ratio(aligned, presentations),
        "explicit_textual_location_cue": has_text_cue,
        "text_cue_alignment_count": aligned if has_text_cue else None,
        "text_cue_alignment_denominator_directional_answers": directional_answers if has_text_cue else None,
        "text_cue_alignment_rate_among_directional_answers": ratio(aligned, directional_answers) if has_text_cue else None,
        "target_alignment_count_among_directional_target_answers": target_matches,
        "target_alignment_denominator_directional_target_answers": target_directional_answers,
        "target_left_predicted_right": target_direction_pairs[("left", "right")],
        "target_right_predicted_left": target_direction_pairs[("right", "left")],
        "target_left_directional_prediction_denominator": target_direction_pairs[("left", "left")] + target_direction_pairs[("left", "right")],
        "target_right_directional_prediction_denominator": target_direction_pairs[("right", "right")] + target_direction_pairs[("right", "left")],
        "target_left_to_predicted_right_rate": ratio(target_direction_pairs[("left", "right")], target_direction_pairs[("left", "left")] + target_direction_pairs[("left", "right")]),
        "target_right_to_predicted_left_rate": ratio(target_direction_pairs[("right", "left")], target_direction_pairs[("right", "right")] + target_direction_pairs[("right", "left")]),
    }


def sham_cue_sensitivity(groups):
    """Same-image cue comparisons with exact logged pixel equality required upstream."""
    side_rows = {}
    for side, regions in (("left", ("tl", "bl")), ("right", ("tr", "br"))):
        predictions = [predicted_direction(group[f"sham_{region}"]) for group in groups for region in regions]
        directions = sum(value in ("left", "right") for value in predictions)
        counts = Counter(value or "other_or_invalid" for value in predictions)
        side_rows[side] = {"n_images": len(groups), "n_presentations": 2 * len(groups),
                           "left_answers": counts["left"], "right_answers": counts["right"],
                           "other_or_invalid_answers": counts["other_or_invalid"],
                           "cue_aligned_answers": counts[side], "directional_answer_denominator": directions,
                           "cue_alignment_rate_among_directional_answers": ratio(counts[side], directions)}
    cross = Counter()
    changed_images = 0
    for group in groups:
        changed = False
        for left_region, right_region in (("tl", "tr"), ("bl", "br")):
            left = predicted_direction(group[f"sham_{left_region}"]) or "other_or_invalid"
            right = predicted_direction(group[f"sham_{right_region}"]) or "other_or_invalid"
            cross[(left, right)] += 1
            changed |= left != right
        changed_images += changed
    return {"n_images": len(groups), "height_matched_left_right_pairs": 2 * len(groups),
            "side_comparison": side_rows,
            "left_cue_answer_by_right_cue_answer": [{"left_cue_answer": left, "right_cue_answer": right, "pair_count": cross[(left, right)]}
                                                      for left in ("left", "right", "other_or_invalid")
                                                      for right in ("left", "right", "other_or_invalid")],
            "unique_images_with_any_direction_category_change": changed_images,
            "interpretation": "All four sham branches repeat identical overview pixels. Height-matched tl/tr and bl/br pairs vary the claimed horizontal direction. This is descriptive sensitivity to a misleading prompt control, not proof of causal word copying."}


def between_condition_switches(groups):
    result = {}
    for new, reference in PRIMARY_CONTRASTS:
        cross = Counter()
        for group in groups:
            for region in REGIONS:
                before = predicted_direction(group[f"{reference}_{region}"])
                after = predicted_direction(group[f"{new}_{region}"])
                if before and after:
                    cross[(before, after)] += 1
        result[f"{reference}_to_{new}"] = {
            "n_images": len(groups), "n_presentations": 4 * len(groups),
            "both_answers_directional_count": sum(cross.values()),
            "left_to_right": cross[("left", "right")], "right_to_left": cross[("right", "left")],
            "left_to_left": cross[("left", "left")], "right_to_right": cross[("right", "right")],
        }
    return result


def cohort_report(groups, bootstrap=False):
    return {
        "n_images": len(groups),
        "direct": transition_metrics(groups, ["direct"]),
        "conditions": {condition: transition_metrics(groups, [f"{condition}_{region}" for region in REGIONS]) for condition in CONDITIONS},
        "per_quadrant": {condition: {region: transition_metrics(groups, [f"{condition}_{region}"]) for region in REGIONS} for condition in CONDITIONS},
        "primary_contrasts": primary_contrasts(groups, bootstrap=bootstrap),
        "direction_diagnostics": {condition: direction_metrics(groups, condition) for condition in CONDITIONS},
        "per_quadrant_direction_diagnostics": {condition: {region: direction_metrics(groups, condition, [region]) for region in REGIONS} for condition in CONDITIONS},
        "paired_condition_direction_switches": between_condition_switches(groups),
    }


def analyze(run: Path, manifest: Path, v1_diagnostics: Path | None = None, v1_run: Path | None = None, allow_smoke=False):
    run, manifest = run.resolve(), manifest.resolve()
    metadata, sources, groups, manifest_count = validate_run(run, manifest, allow_smoke=allow_smoke)
    minimum = metadata["config"].get("analysis", {}).get("minimum_complete_examples_for_primary_report", 1)
    primary_complete = len(groups) >= minimum
    print(f"Validated complete run: {len(groups)} images x 17 actions. Bootstrapping paired image contrasts...", flush=True)
    result = {
        "schema_version": 1, "status": "complete" if primary_complete else "engineering_smoke",
        "completed_primary_cohort": primary_complete, "minimum_primary_images": minimum,
        "run_fingerprint": metadata["fingerprint"],
        "manifest_sha256": metadata["manifest_sha256"], "records_sha256": sha256_file(run / "records.jsonl"),
        "analysis_script_sha256": sha256_file(Path(__file__)), "recorded_source_code_sha256": metadata["code_sha256"],
        "model": metadata["model"],
        "validated_condition_pixel_integrity": {
            "status": "passed", "n_images": len(groups), "scope": "Relations among recorded image RGB hashes and realized prompts/geometry; source files are independently byte-verified",
            "same_overview_hash_across_all_seventeen_actions": True,
            "named_neutral_frame_same_crop_hash_per_region": True,
            "all_sham_second_image_hashes_equal_direct_overview": True,
            "named_sham_same_realized_prompt_per_region": True,
            "claimed_actual_boxes_and_additional_view_kinds_match_design": True,
            "limitation": "This checks logged intervention identities rather than re-executing the image processor. Actual grids and image sizes can differ between a sham overview and a true crop; no equal-compute claim is made.",
        },
        "coverage": {"planned_images": len(groups), "complete_images": len(groups), "manifest_examples": manifest_count,
                     "records": len(groups) * 17, "actions_per_image": 17, "failed_or_excluded_records": 0,
                     "unique_source_images": True, "image_byte_hashes_verified": True, "decoded_exact_duplicates": 0},
        "estimand": "Mean across images of the mean correctness over four quadrants in each condition; equal image weights and uniform quadrant weights",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": SEED, "unit": "original_image_id", "interval": "95% percentile",
                      "pairing": "all condition/quadrant outcomes retained together", "multiple_comparison_adjustment": "none", "p_values": "not computed"},
        "main": cohort_report(groups, bootstrap=True),
        "question_lexical_subgroups": {}, "target_answer_subgroups_secondary": {},
        "cost_by_action_descriptive": {}, "v1_outcome_selected_subset_exploratory": None,
        "format_diagnostics_by_action": {},
        "v1_fresh_replication_descriptive": None,
        "direction_definitions": {
            "question_regex_casefold": DIRECTION_PATTERN.pattern,
            "direction_answer_rule": "parse_valid and conservative normalized predicted_answer exactly left or right",
            "target_group_rule": "conservative normalized target_answer exactly left, right, or other; analysis only",
            "normalization": "strip, casefold, remove trailing .!? then collapse whitespace; never rescoring",
            "location_side_map": {"tl": "left", "bl": "left", "tr": "right", "br": "right"},
            "text_cue_conditions": ["named", "sham", "frame"],
            "neutral_alignment": "Neutral has no explicit location cue; only region-side correspondence is descriptive",
        },
        "caveats": CAVEATS,
    }
    lexical_groups = {"contains_direction_word": [], "no_direction_word": []}
    target_groups = {"left": [], "right": [], "other": []}
    for group in groups:
        direct = group["direct"]
        lexical = "contains_direction_word" if DIRECTION_PATTERN.search(direct["question"].casefold()) else "no_direction_word"
        lexical_groups[lexical].append(group)
        target = normalize(direct["target_answer"])
        target_groups[target if target in ("left", "right") else "other"].append(group)
    for label, subgroup in lexical_groups.items():
        result["question_lexical_subgroups"][label] = {"selection": "prespecified secondary question-text subgroup",
                                                      **cohort_report(subgroup, bootstrap=True)}
    result["sham_left_right_cue_sensitivity_lexical_subgroup"] = sham_cue_sensitivity(lexical_groups["contains_direction_word"])
    for label, subgroup in target_groups.items():
        result["target_answer_subgroups_secondary"][label] = {"selection": "secondary label-conditioned descriptive analysis only",
                                                              **cohort_report(subgroup, bootstrap=False)}
    for action in ACTIONS:
        action_rows = [group[action] for group in groups]
        result["format_diagnostics_by_action"][action] = {
            "records": len(action_rows),
            "invalid_answers": sum(not row["parse_valid"] for row in action_rows),
            "truncated_generations": sum(row["generation_truncated"] for row in action_rows),
            "generated_tokens": describe_numbers([row["generated_tokens"] for row in action_rows]),
            "interpretation": "Original recorded parse and truncation flags; no answers or correctness labels are changed",
        }
        elapsed = [group[action]["elapsed_s"] + (0 if action == "direct" else group["direct"]["elapsed_s"]) for group in groups]
        memory = [group[action]["peak_memory_gib"] if action == "direct" else max(group[action]["peak_memory_gib"], group["direct"]["peak_memory_gib"]) for group in groups]
        result["cost_by_action_descriptive"][action] = {"n_images": len(groups), "full_path_latency_s": describe_numbers(elapsed),
                                                       "sequential_peak_memory_gib": describe_numbers(memory)}
    if v1_diagnostics:
        previous = json.loads(v1_diagnostics.read_text(encoding="utf-8"))
        if previous.get("status") != "complete":
            raise ValueError("Optional v1 diagnostics must describe a completed run")
        pinned_v1 = metadata["config"].get("protocol", {}).get("v1_run_fingerprint")
        if pinned_v1 and previous.get("run_fingerprint") != pinned_v1:
            raise ValueError("Optional v1 diagnostic fingerprint differs from the frozen v1 reference")
        selected_ids = previous["crop_exclusive_rescues"]["example_ids"]
        if selected_ids:
            unique_strings(selected_ids, "Optional v1 selected IDs")
        if previous["crop_exclusive_rescues"].get("count") != len(selected_ids):
            raise ValueError("Optional v1 selected count disagrees with IDs")
        if set(selected_ids) - set(metadata["example_ids"]):
            raise ValueError("Optional v1 selected IDs are not all in the completed v2 run")
        subset = [group for group in groups if group["direct"]["example_id"] in set(selected_ids)]
        result["v1_outcome_selected_subset_exploratory"] = {
            "selection": "Previous-v1 crop-exclusive rescues selected by outcome; exploratory description only",
            "source_file_sha256": sha256_file(v1_diagnostics), "v1_run_fingerprint": previous.get("run_fingerprint"),
            "selected_example_ids": selected_ids, **cohort_report(subset, bootstrap=False),
        }
    if v1_run:
        result["v1_fresh_replication_descriptive"] = compare_v1(v1_run.resolve(), metadata, sources, groups)
    return result


def compare_v1(v1_run, metadata, sources, groups):
    old = json.loads((v1_run / "run.json").read_text(encoding="utf-8"))
    identity = {key: old[key] for key in IDENTITY_KEYS}
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != old.get("fingerprint"):
        raise ValueError("Optional v1 run fingerprint mismatch")
    pinned_v1 = metadata["config"].get("protocol", {}).get("v1_run_fingerprint")
    if pinned_v1 and old["fingerprint"] != pinned_v1:
        raise ValueError("Optional v1 fingerprint differs from the frozen v1 reference")
    if old["manifest_sha256"] != metadata["manifest_sha256"] or old["model"] != metadata["model"]:
        raise ValueError("Optional v1 run must use the same manifest and model revision")
    unique_strings(old["example_ids"], "Optional v1 planned IDs")
    if set(metadata["example_ids"]) - set(old["example_ids"]):
        raise ValueError("Some v2 planned examples are absent from the v1 run")
    old_actions = old["config"]["actions"]
    unique_strings(old_actions, "Optional v1 actions")
    expected = {(example, action) for example in old["example_ids"] for action in old_actions}
    by_key = {}
    for row in read_jsonl(v1_run / "records.jsonl"):
        key = (row.get("example_id"), row.get("action"))
        if key in by_key or key not in expected or row.get("status") != "ok":
            raise ValueError("Optional v1 run contains invalid/duplicate/undeclared records")
        if not isinstance(row.get("correct"), bool) or not isinstance(row.get("parse_valid"), bool):
            raise ValueError("Optional v1 record has invalid grades")
        if (row["parse_valid"] and (not isinstance(row.get("predicted_answer"), str) or not row["predicted_answer"].strip())) or (not row["parse_valid"] and row["correct"]):
            raise ValueError("Optional v1 record has inconsistent parse fields")
        if row["example_id"] in sources:
            source = sources[row["example_id"]]
            if (row.get("image_id") != source["image_id"] or row.get("question") != source["question"]
                    or row.get("target_answer") != source["answer"]):
                raise ValueError("Optional v1 records disagree with the current manifest")
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError("Optional v1 run is incomplete")
    methods = {}
    for new_action, old_action in [("direct", "direct"), *((f"named_{r}", f"crop_{r}") for r in REGIONS)]:
        if old_action not in old_actions:
            raise ValueError("Optional v1 run lacks required direct/crop actions")
        pairs = [(group[new_action], by_key[(group["direct"]["example_id"], old_action)]) for group in groups]
        valid_pairs = [(new, previous) for new, previous in pairs if new["parse_valid"] and previous["parse_valid"]]
        matches = sum(normalize(new.get("predicted_answer")) == normalize(previous.get("predicted_answer")) for new, previous in valid_pairs)
        methods[new_action] = {"v1_action": old_action, "n_images": len(groups), "both_parse_valid_count": len(valid_pairs),
                               "normalized_answer_match_count": matches, "normalized_answer_match_rate_among_valid_pairs": ratio(matches, len(valid_pairs)),
                               "pairs_not_both_parse_valid": len(pairs) - len(valid_pairs),
                               "correctness_agreement_count": sum(new["correct"] == previous["correct"] for new, previous in pairs)}
    return {"v1_run_fingerprint": old["fingerprint"], "v1_records_sha256": sha256_file(v1_run / "records.jsonl"),
            "role": "fresh direct/named replication against archived v1; descriptive, not pooled or used for selection", "methods": methods}


def pct(value):
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def cohort_markdown(cohort, include_quadrants=True, include_contrasts=True):
    n = cohort["n_images"]
    lines = [f"Independent image units: **{n}**. Direct accuracy: **{pct(cohort['direct']['expected_accuracy'])}**. "
             f"Each condition has {4 * n} presentations, with four paired presentations per image.", "",
             "| Condition | Expected accuracy, uniform quadrant | Delta vs direct, pp | Fixes / presentations | Harms / presentations | Images with any fix / any harm | Invalid parses |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for condition in CONDITIONS:
        row = cohort["conditions"][condition]
        delta = f"{row['delta_vs_direct_pp']:+.2f}" if row["delta_vs_direct_pp"] is not None else "unavailable"
        lines.append(f"| {condition} | {pct(row['expected_accuracy'])} | {delta} | {row['fix_presentations']} / {row['n_presentations']} | "
                     f"{row['harm_presentations']} / {row['n_presentations']} | {row['unique_images_with_any_fix']} / {row['unique_images_with_any_harm']} | {row['invalid_parse_presentations']} |")
    if include_contrasts:
        lines.extend(["", "| Paired contrast | Images | Difference, pp | 95% paired image-bootstrap interval, pp |", "| --- | ---: | ---: | ---: |"])
        for name, row in cohort["primary_contrasts"].items():
            interval = row["delta_ci_95_pp"]
            shown = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "not estimated"
            delta = f"{row['delta_expected_accuracy_pp']:+.2f}" if row["delta_expected_accuracy_pp"] is not None else "unavailable"
            lines.append(f"| {name} | {n} | {delta} | {shown} |")
    if include_quadrants:
        lines.extend(["", "| Condition | Quadrant | Images | Accuracy | Fixes / harms |", "| --- | --- | ---: | ---: | ---: |"])
        for condition in CONDITIONS:
            for region in REGIONS:
                row = cohort["per_quadrant"][condition][region]
                lines.append(f"| {condition} | {region} | {n} | {pct(row['expected_accuracy'])} | {row['fix_presentations']} / {row['harm_presentations']} |")
    lines.extend(["", "Direction switches are relative to the shared direct answer and count presentations. Cue alignment uses only parsed left/right answers; the denominator is explicit.", "",
                  "| Condition | Direct left→right | Direct right→left | Cue-aligned / directional answers | Cue-alignment rate |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for condition in CONDITIONS:
        row = cohort["direction_diagnostics"][condition]
        alignment = f"{row['text_cue_alignment_count']} / {row['text_cue_alignment_denominator_directional_answers']}" if row["explicit_textual_location_cue"] else "no textual cue"
        lines.append(f"| {condition} | {row['direct_left_to_followup_right']} | {row['direct_right_to_followup_left']} | {alignment} | {pct(row['text_cue_alignment_rate_among_directional_answers'])} |")
    return lines


def markdown(result):
    lines = ["# LookAgain v2: direction-control experiment", "",
             f"Status: **{result['status']}**. " + ("The protocol's primary cohort size is satisfied." if result["completed_primary_cohort"] else
                 f"Engineering smoke only: below the required {result['minimum_primary_images']} images; this is not a completed primary result."), "",
             f"Complete planned cohort: **{result['coverage']['complete_images']} unique images**, all **17 actions** per image. "
             "The original correctness labels are retained, including invalid parses as incorrect.", "",
             f"- Run fingerprint: `{result['run_fingerprint']}`.", f"- Manifest SHA-256: `{result['manifest_sha256']}`.",
             f"- Records SHA-256: `{result['records_sha256']}`.", f"- Analysis script SHA-256: `{result['analysis_script_sha256']}`.", "",
             "Logged intervention integrity passed: all overviews match; named/neutral/frame use the same crop pixels per quadrant; "
             "every sham repeats the direct overview exactly; named and sham prompts match. "
             "Source-image bytes were also verified. These are recorded-hash checks, not an independent image-processor rerun.", "",
             "## Main paired analysis", "",
             "The per-condition estimate averages all four quadrant outcomes per image, then averages images. "
             "The sample size for uncertainty is the number of images, not the number of presentations.", ""]
    lines.extend(cohort_markdown(result["main"]))
    lines.extend(["", "## Prespecified secondary question-text subgroup", "",
                  f"Casefolded question regex: `{DIRECTION_PATTERN.pattern}`. This grouping does not use target answers. "
                  "Both matching and nonmatching groups are shown; subgroup intervals remain unadjusted.", ""])
    for label, cohort in result["question_lexical_subgroups"].items():
        lines.extend([f"### {label}", "", *cohort_markdown(cohort, include_quadrants=False), ""])
    sensitivity = result["sham_left_right_cue_sensitivity_lexical_subgroup"]
    lines.extend(["### Identical-pixel sham: left versus right cue", "",
                  sensitivity["interpretation"], "", f"Lexical-subgroup images: {sensitivity['n_images']}; "
                  f"height-matched left/right pairs: {sensitivity['height_matched_left_right_pairs']}. "
                  "These pairs share images and are not independent sampling units.", "",
                  "| Claimed cue side | Presentations | Left answers | Right answers | Other/invalid | Cue-aligned / directional | Alignment rate |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for side, row in sensitivity["side_comparison"].items():
        lines.append(f"| {side} | {row['n_presentations']} | {row['left_answers']} | {row['right_answers']} | {row['other_or_invalid_answers']} | "
                     f"{row['cue_aligned_answers']} / {row['directional_answer_denominator']} | {pct(row['cue_alignment_rate_among_directional_answers'])} |")
    lines.extend(["", "| Answer with left cue | Answer with right cue | Height-matched pair count |", "| --- | --- | ---: |"])
    for row in sensitivity["left_cue_answer_by_right_cue_answer"]:
        lines.append(f"| {row['left_cue_answer']} | {row['right_cue_answer']} | {row['pair_count']} |")
    lines.extend(["## Secondary target-answer groups", "", "Targets are grouped as normalized left, right or other for analysis only. "
                  "These label-conditioned descriptions have no inferential intervals.", ""])
    for label, cohort in result["target_answer_subgroups_secondary"].items():
        lines.extend([f"### Target: {label}", "", *cohort_markdown(cohort, include_quadrants=False, include_contrasts=False), ""])
    subset = result["v1_outcome_selected_subset_exploratory"]
    if subset is not None:
        lines.extend(["## Exploratory subset selected by v1 outcomes", "",
                      "These images were selected because of previous crop-exclusive outcomes. This is not independent confirmation; "
                      "no intervals are estimated for this subset.", "", *cohort_markdown(subset, include_quadrants=True, include_contrasts=False), ""])
    replication = result["v1_fresh_replication_descriptive"]
    if replication:
        lines.extend(["## Fresh replication versus archived v1", "", replication["role"], "",
                      "| Fresh action | Archived action | Matched normalized answers / both parsed | Match rate | Correctness agreement / images |",
                      "| --- | --- | ---: | ---: | ---: |"])
        for action, row in replication["methods"].items():
            lines.append(f"| {action} | {row['v1_action']} | {row['normalized_answer_match_count']} / {row['both_parse_valid_count']} | "
                         f"{pct(row['normalized_answer_match_rate_among_valid_pairs'])} | {row['correctness_agreement_count']} / {row['n_images']} |")
        lines.append("")
    lines.extend(["## Output format and truncation", "",
                  "Counts use the recorded parse and truncation flags. Token counts must be positive integers within the configured generation limit. "
                  "These checks and diagnostics do not alter any answer or correctness label.", "",
                  "| Action | Records | Invalid answers | Truncated generations | Generated tokens, mean / min / max |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for action, row in result["format_diagnostics_by_action"].items():
        tokens = row["generated_tokens"]
        lines.append(f"| {action} | {row['records']} | {row['invalid_answers']} | {row['truncated_generations']} | "
                     f"{tokens['mean']:.2f} / {tokens['min']} / {tokens['max']} |")
    lines.extend(["", "## Descriptive full-path timing", "",
                  "Each follow-up includes the shared direct invocation plus that follow-up's incremental elapsed time. "
                  "Actions are shown separately; this is not a pooled deployment-cost claim.", "",
                  "| Fixed action | Images | Mean / median / p95 seconds | Maximum sequential allocated GiB |",
                  "| --- | ---: | ---: | ---: |"])
    for action, row in result["cost_by_action_descriptive"].items():
        timing = row["full_path_latency_s"]
        lines.append(f"| {action} | {row['n_images']} | {timing['mean']:.3f} / {timing['median']:.3f} / {timing['p95']:.3f} | {row['sequential_peak_memory_gib']['max']:.2f} |")
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {caveat}" for caveat in result["caveats"])
    lines.extend(["", "The JSON also includes per-quadrant direction diagnostics, conditional fix/harm denominators and "
                  "paired direction switches between conditions. It contains no learned controller or selected best quadrant.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Exact run manifest beside its saved images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v1-diagnostics", type=Path)
    parser.add_argument("--v1-run", type=Path, help="Optional complete archived v1 run for descriptive fresh-replication checks")
    parser.add_argument("--allow-smoke", action="store_true", help="Allow a complete smaller declared run, labeled engineering_smoke rather than a primary result")
    args = parser.parse_args()
    result = analyze(args.run, args.manifest, args.v1_diagnostics, args.v1_run, allow_smoke=args.allow_smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(markdown(result), encoding="utf-8")
    print(f"Wrote {result['status']} paired direction-control analysis to {args.output}")


if __name__ == "__main__":
    main()
