"""Strict completed-run analysis of annotation-privileged ROI availability.

CPU only: reconstructs image/chat provenance and grades, never loads a VLM.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT / "src"))
from evidence_availability_core import ACTIONS, BUDGETS, FIDELITIES, build_request, parse_action, score_response
from lookagain.runner import code_digest, sha256_file
from tatdqa_metrics import score_official, UPSTREAM_COMMIT, UPSTREAM_SHA256
from analyze_native_detail import canonical_hash, read_json, read_jsonl, require, number, describe
from analyze_history_context import paired_contrast

METRICS = ("official_em", "conservative_text_em", "official_f1", "anls")
BINARY_METRICS = ("official_em", "conservative_text_em")
IDENTITY_KEYS = ("config", "manifest_sha256", "code_sha256", "model", "runtime", "role", "example_ids")
PRIMARY = "native_256_minus_degraded_256"
INTERACTION = "detail_256_minus_detail_1024"
SOURCE_PINS = {
    "train_json_sha256": "3025b4ca9c3c87f5ecd2c504371e1ee92929ef1f31d8a3ac38bc3a5376582fab",
    "train_zip_sha256": "412dde804ec9d5888b888a9d7179f2157bc2679b92379f9804e6f2062e7237d9",
    "source_main_manifest_sha256": "757c3a4d024ac726f701c8732424e3457bfb292d224f67e2de22201ac641adc5",
    "source_smoke_manifest_sha256": "ac4f35d9edf4c6e6a7b70225668fc89172a618f3eaecaa661009269c2c109564",
    "source_quality_mask_sha256": "a621e6124f866ec4a1d0b882dc03cd5a10fe3bcfd8e567574c65934c76f9db90",
}


def utc(value):
    require(isinstance(value, str), "Timestamp must be a string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "Timestamp must include a timezone")
    return result


def finite_tree(value, path="configuration"):
    """JSON parsers otherwise accept NaN/Infinity, including nested parameters."""
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, path + "/" + str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            finite_tree(child, path + "/" + str(i))
    elif isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite {path}")


def cost(group, action, field="elapsed_s", mode="standalone"):
    """All requests are standalone; hypothetical decision-state cost adds direct."""
    require(mode in ("standalone", "decision_state"), "Unknown cost perspective")
    kind, budget = parse_action(action)
    value = group[action][field]
    if mode == "standalone" or kind not in FIDELITIES:
        return value
    initial = group[f"direct_{budget}"][field]
    return max(value, initial) if field in ("peak_memory_gib", "peak_reserved_gib") else value + initial


def summarize_action(groups, action):
    rows = [g[action] for g in groups]
    result = {
        "n_source_reports": len(groups), "n_presentations": len(rows),
        "metrics": {m: statistics.fmean(r[m] for r in rows) for m in METRICS},
        "invalid_answers": sum(not r["parse_valid"] for r in rows),
        "truncated_generations": sum(r["generation_truncated"] for r in rows),
        "invocation_latency_s": describe([r["elapsed_s"] for r in rows]),
    }
    for mode in ("standalone", "decision_state"):
        result[mode + "_latency_s"] = describe([cost(g, action, mode=mode) for g in groups])
        result[mode + "_peak_allocated_gib"] = describe([cost(g, action, "peak_memory_gib", mode) for g in groups])
        result[mode + "_peak_reserved_gib"] = describe([cost(g, action, "peak_reserved_gib", mode) for g in groups])
    result["tokens"] = {field: {mode: describe([cost(g, action, field, mode) for g in groups])
                               for mode in ("standalone", "decision_state")}
                       for field in ("input_tokens", "visual_tokens", "generated_tokens")}
    return result


def transitions(groups, positive, negative, metric):
    require(metric in BINARY_METRICS, "Transitions require a binary metric")
    before = [g[negative][metric] for g in groups]
    after = [g[positive][metric] for g in groups]
    require(all(x in (0., 1.) for x in before + after), "Nonbinary transition score")
    correct = sum(x == 1 for x in before)
    wrong = len(groups) - correct
    fixes = sum(b == 0 and a == 1 for b, a in zip(before, after))
    harms = sum(b == 1 and a == 0 for b, a in zip(before, after))
    return {"metric": metric, "positive": positive, "negative": negative,
            "n_source_reports": len(groups), "fixes": fixes, "harms": harms,
            "negative_correct": correct, "negative_wrong": wrong,
            "correct_to_correct": correct - harms, "wrong_to_wrong": wrong - fixes,
            "fix_rate_among_negative_wrong": fixes / wrong if wrong else None,
            "harm_rate_among_negative_correct": harms / correct if correct else None,
            "fix_fraction_all_reports": fixes / len(groups), "harm_fraction_all_reports": harms / len(groups)}


def pair_diagnostics(groups, positive, negative):
    pairs = [(g[positive], g[negative]) for g in groups]
    changed = [i for i, (a, b) in enumerate(pairs) if a["predicted_answer"] != b["predicted_answer"]]
    invalid_raw_changes = [i for i, (a, b) in enumerate(pairs)
                           if a["response"] != b["response"] and not (a["parse_valid"] and b["parse_valid"])]
    audit_indices = sorted(set(changed) | set(invalid_raw_changes))
    return {"n_source_reports": len(groups),
            "raw_response_changed": sum(a["response"] != b["response"] for a, b in pairs),
            "parsed_answer_changed": len(changed), "parsed_answer_changed_source_indices": changed,
            "raw_response_changed_with_invalid_parse_source_indices": invalid_raw_changes,
            "semantic_audit_required": len(audit_indices), "semantic_audit_required_source_indices": audit_indices,
            "both_parse_valid": sum(a["parse_valid"] and b["parse_valid"] for a, b in pairs),
            "metric_equal": {m: sum(a[m] == b[m] for a, b in pairs) for m in METRICS},
            "audit_scope": "Review every parsed-answer change, including equal-grade pairs, plus raw-response differences when either parse is invalid; no semantic judgments are inferred here."}


def summarize(groups, config, bootstrap=None):
    require(bool(groups) and all(set(g) == set(ACTIONS) for g in groups), "Complete ten-action groups required")
    analysis = config["analysis"]
    require(analysis["primary_metric"] == "official_em", "Unexpected primary metric")
    samples = analysis.get("bootstrap_samples", 10000) if bootstrap is None else bootstrap
    seed = analysis.get("bootstrap_seed", 20260926)
    require(type(samples) is int and samples > 0 and type(seed) is int, "Invalid bootstrap parameters")
    n = len(groups)
    rng = random.Random(seed)
    draws = np.asarray([[rng.randrange(n) for _ in range(n)] for _ in range(samples)], dtype=np.int32)
    vectors = {m: {a: [g[a][m] for g in groups] for a in ACTIONS} for m in METRICS}
    detail_vectors = {m: {str(b): [x - y for x, y in zip(vectors[m][f"native_{b}"], vectors[m][f"degraded_{b}"])]
                          for b in BUDGETS} for m in METRICS}
    contrasts = {}
    for budget in BUDGETS:
        for kind in FIDELITIES:
            positive = f"{kind}_{budget}"
            negatives = ([f"degraded_{budget}"] if kind == "native" else []) + [f"direct_{budget}", "highres"]
            for negative in negatives:
                name = positive + "_minus_" + negative
                contrasts[name] = {"role": "primary; official EM only" if name == PRIMARY else "secondary",
                                   "positive": positive, "negative": negative,
                                   "metrics": {m: paired_contrast(vectors[m][positive], vectors[m][negative], draws) for m in METRICS}}
        name = f"highres_minus_direct_{budget}"
        contrasts[name] = {"role": "secondary", "positive": "highres", "negative": f"direct_{budget}",
                           "metrics": {m: paired_contrast(vectors[m]["highres"], vectors[m][f"direct_{budget}"], draws) for m in METRICS}}
    contrasts[INTERACTION] = {"role": "secondary interaction",
        "definition": "(native_256 - degraded_256) - (native_1024 - degraded_1024), formed within source report",
        "metrics": {m: paired_contrast(detail_vectors[m]["256"], detail_vectors[m]["1024"], draws) for m in METRICS}}
    detail = {str(b): {m: paired_contrast(detail_vectors[m][str(b)], [0.] * n, draws) for m in METRICS} for b in BUDGETS}
    threshold = analysis.get("candidate_gate", {}).get("detail_effect_threshold", .02)
    number(threshold, "candidate threshold")
    primary = detail["256"]["official_em"]
    checks = {"detail_mean_at_least_threshold": primary["difference"] >= threshold,
              "detail_ci95_lower_positive": primary["ci95"][0] > 0,
              "native_mean_at_least_matched_direct": statistics.fmean(vectors["official_em"]["native_256"]) >= statistics.fmean(vectors["official_em"]["direct_256"])}
    summaries = {a: summarize_action(groups, a) for a in ACTIONS}
    return {"primary_contrast": PRIMARY, "primary_metric": "official_em", "conditions": summaries, "actions": summaries,
            "contrasts": contrasts, "detail_effects": detail,
            "per_source_action_scores": vectors,
            "paired_native_degraded_transitions": {str(b): {m: transitions(groups, f"native_{b}", f"degraded_{b}", m) for m in BINARY_METRICS} for b in BUDGETS},
            "changes_vs_matched_direct": {a: {m: transitions(groups, a, f"direct_{parse_action(a)[1]}", m) for m in BINARY_METRICS}
                                          for a in ACTIONS if parse_action(a)[0] in FIDELITIES},
            "paired_native_degraded_response_agreement": {str(b): pair_diagnostics(groups, f"native_{b}", f"degraded_{b}") for b in BUDGETS},
            "candidate_gate": {"metric": "official_em", "budget": 256, "detail_effect_threshold": threshold,
                "checks": checks, "quantitative_rule_met": all(checks.values()),
                "semantic_audit_status": "pending_separate_artifact", "automatic_controller_training": False,
                "controller_ready": False,
                "interpretation": "Quantitative evidence-availability screen only. Privileged ROI, metric robustness and a separate semantic audit preclude an automatic policy/training conclusion."},
            "privileged_localization": {"label": "Reference-annotation-guided ROI; not a learned or deployable selection policy",
                "target_text_in_language_prompt": False, "annotation_location_is_privileged_information": True,
                "oracle_over_actions_reported": False, "new_unseen_report_confirmation": False},
            "bootstrap": {"samples": samples, "seed": seed, "unit": "original_source_report", "n_source_reports": n,
                "retains_all_budgets_and_fidelities": True, "interval": "95% percentile; secondary intervals unadjusted",
                "resampling": "Shared random.Random(seed).randrange(n) report draws; interaction formed within report before resampling."}}


def validate_roi_geometry(audit):
    """Independently recompute the frozen geometry without PDF parser imports."""
    evidence, geometry = audit["evidence"], audit["pdf_geometry"]
    page, media, crop = evidence["ocr_page_bbox"], geometry["mediabox"], geometry["cropbox"]
    def contains(outer, inner):
        return (isinstance(outer, list) and isinstance(inner, list) and len(outer) == len(inner) == 4
                and all(type(x) in (int, float) and math.isfinite(x) for x in outer + inner)
                and outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3])
    require(geometry["page_count"] == 1 and geometry["rotation"] == 0 and geometry["user_unit"] == 1,
            "Unsupported PDF geometry in audit")
    require(contains(media, crop) and contains(page, page), "Invalid PDF/OCR boxes")
    ow, oh, cw, ch = page[2]-page[0], page[3]-page[1], crop[2]-crop[0], crop[3]-crop[1]
    require(abs(ow * ch - oh * cw) <= cw + ch, "OCR/CropBox aspect mismatch")
    width, height = audit["source_size"]
    require(type(width) is int and type(height) is int and min(width, height) >= 256, "Invalid ROI source dimensions")
    boxes = [w["bbox"] for w in evidence["selected_words"]]
    require(bool(boxes) and all(contains(page, b) for b in boxes), "Invalid selected word boxes")
    raw = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    mx, my = width/(media[2]-media[0]), height/(media[3]-media[1])
    sx, sy = cw/ow*mx, ch/oh*my
    dx, dy = (crop[0]-media[0])*mx, (media[3]-crop[3])*my
    box = [math.floor(dx+(raw[0]-page[0])*sx), math.floor(dy+(raw[1]-page[1])*sy),
           math.ceil(dx+(raw[2]-page[0])*sx), math.ceil(dy+(raw[3]-page[1])*sy)]
    require(contains([0, 0, width, height], box), "Mapped evidence outside render")
    median_height = statistics.median((b[3]-b[1])*sy for b in boxes)
    margin = 2 * median_height
    rw, rh = min(width, max(256, math.ceil(box[2]-box[0]+2*margin))), min(height, max(256, math.ceil(box[3]-box[1]+2*margin)))
    left = min(width-rw, max(0, math.floor((box[0]+box[2]-rw)/2)))
    top = min(height-rh, max(0, math.floor((box[1]+box[3]-rh)/2)))
    expected = [left, top, left+rw, top+rh]
    require(expected == audit["roi_pixels"] and contains(expected, box) and rw*rh/(width*height) <= .25, "ROI reconstruction differs")
    construction = audit["construction"]
    for key, value in (("raw_word_bbox_ocr", raw), ("raw_word_bbox_pixels", box),
                       ("median_word_height_pixels", median_height), ("context_margin_pixels", margin),
                       ("minimum_roi_side_pixels", 256), ("roi_page_area_fraction", rw*rh/(width*height))):
        require(construction.get(key) == value, f"ROI construction differs: {key}")
    transform = construction["transform"]
    for key, value in (("scale_xy", [sx, sy]), ("offset_xy", [dx, dy]),
                       ("pdf_to_render_scale_xy", [mx, my]), ("ocr_origin_xy", page[:2])):
        require(transform.get(key) == value, f"ROI transform differs: {key}")


def validate_preparation(manifest, metadata, sources, role):
    """Bind source-only eligibility and reconstruct the ROI from audited boxes.

    Original large ZIP/PDF bytes are not reread: their exact hashes are pinned
    by the preparation script and its archived audit, not independently rerun.
    """
    roi_hash = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    normalize_span = lambda value: " ".join(value.casefold().split())
    root, pins = Path(manifest).resolve().parent, metadata["config"]["protocol"]
    for filename, key in (("selection_metadata.json", "preparation_metadata_sha256"), ("roi_audit.jsonl", "roi_audit_sha256")):
        require(sha256_file(root / filename) == pins[key], f"Preparation binding mismatch: {filename}")
    preparation = read_json(root / "selection_metadata.json")
    finite_tree(preparation, "preparation")
    require(preparation.get("stage") == "evidence_availability_development", "Wrong preparation stage")
    for key, expected in (("selection_uses_model_outputs", False), ("same_prior_development_reports", True),
                          ("held_out_claim", False), ("privileged_annotation_roi", True), ("source_images_copied_byte_identically", True)):
        require(preparation.get(key) is expected, f"Invalid preparation declaration: {key}")
    require(preparation.get("pins") == SOURCE_PINS, "Original source pins differ")
    require(preparation.get("preparation_script_sha256") == sha256_file(PROJECT / "experiments/prepare_evidence_roi.py"), "Preparation script binding differs")
    require(utc(preparation["created_utc"]) <= utc(metadata["created_utc"]), "Preparation postdates inference")
    require(preparation.get(role + "_manifest_sha256") == metadata["manifest_sha256"], "Preparation manifest differs")
    require(preparation.get("roi_audit_sha256") == pins["roi_audit_sha256"], "Preparation audit binding differs")
    decisions = preparation["decisions"]
    require(len({d["example_id"] for d in decisions}) == len(decisions), "Duplicate eligibility decision")
    eligible = [d for d in decisions if d["cohort"] == role and d["eligible"] is True]
    require([d["example_id"] for d in eligible] == [s["example_id"] for s in sources], "Eligibility order/cohort differs")
    require(preparation["counts"][role]["eligible_sources"] == len(sources), "Eligibility count differs")
    audit_rows = read_jsonl(root / "roi_audit.jsonl")
    require(len({a["example_id"] for a in audit_rows}) == len(audit_rows), "Duplicate ROI audit")
    audits = {a["example_id"]: a for a in audit_rows}
    for source, decision in zip(sources, eligible):
        finite_tree(source, "source")
        audit = audits.get(source["example_id"])
        require(isinstance(audit, dict), "Missing ROI audit")
        finite_tree(audit, "ROI audit")
        require(roi_hash(audit) == source["roi_provenance_sha256"] == decision["roi_provenance_sha256"], "ROI provenance hash differs")
        require(source.get("roi_privileged") is True and audit.get("privileged_annotation_roi") is True
                and audit.get("model_outcomes_used") is False, "ROI privilege/outcome declaration differs")
        require(source.get("source_quality_category") in ("none", "mapping_only")
                and audit.get("source_quality_category") == source["source_quality_category"]
                and decision.get("source_quality_category") == source["source_quality_category"], "Source quality eligibility differs")
        for audit_key, source_key in (("source_id", "source_id"), ("question", "question"), ("reference_answer", "answer"),
                                      ("source_image_sha256", "image_sha256"), ("source_pdf_sha256", "source_pdf_sha256"),
                                      ("source_manifest_sha256", "source_manifest_sha256")):
            require(audit.get(audit_key) == source[source_key], f"ROI audit/source differs: {audit_key}")
        require(audit.get("cohort") == role and source["source_manifest_sha256"] == SOURCE_PINS[f"source_{role}_manifest_sha256"], "ROI source cohort differs")
        require(audit["roi_pixels"] == decision["roi_pixels"] == source["roi_pixels"], "ROI geometry binding differs")
        evidence = audit["evidence"]
        require(normalize_span(evidence["selected_text"]) == normalize_span(source["answer"])
                and evidence["unsafe_boundary_cut"] is False, "ROI reference alignment differs")
        validate_roi_geometry(audit)
        with Image.open(root / source["image_path"]) as image:
            require(list(image.size) == audit["source_size"] == source["pixel_dimensions"], "ROI source-size binding differs")
    return preparation


def validate_run(run, manifest, config_path, *, expected_role="main"):
    # Import lazily so estimand unit tests need no runner initialization.
    from evidence_availability import BASE_SOURCE_SHA256, experiment_digest, validate_config, validate_manifest
    run, manifest, config_path = Path(run).resolve(), Path(manifest).resolve(), Path(config_path).resolve()
    metadata = read_json(run / "run.json")
    finite_tree(metadata)
    require(all(k in metadata for k in IDENTITY_KEYS), "Incomplete run identity")
    require(canonical_hash({k: metadata[k] for k in IDENTITY_KEYS}) == metadata.get("fingerprint"), "Run fingerprint mismatch")
    config = metadata["config"]
    require(config == read_json(config_path), "Run config differs from supplied locked config")
    validate_config(config)
    require(config["actions"] == ACTIONS, "Wrong action order")
    require(metadata.get("experiment") == "evidence_availability_v5" and metadata.get("development_only") is True
            and metadata.get("label_privileged_localizer") is True, "Unexpected experiment or missing privileged-localizer declaration")
    require(expected_role in ("main", "smoke") and metadata.get("role") == expected_role, "Wrong panel role")
    require(metadata.get("base_code_sha256") == code_digest() == BASE_SOURCE_SHA256, "Frozen base source differs")
    pins = config["protocol"]
    require(metadata["code_sha256"] == experiment_digest() == pins["inference_code_sha256"], "V5 inference code differs")
    require(metadata["model"] == {"model_id": config["model_id"], "revision": pins["model_revision"]}, "Model identity mismatch")
    for key in ("rules_locked_at_utc", "main_execution_locked_at_utc"):
        require(utc(pins[key]) <= utc(metadata["created_utc"]), f"Run predates {key}")
    for key, value in metadata["runtime"].items():
        require(metadata.get(key) == value, f"Inconsistent runtime field {key}")
    require(sha256_file(manifest) == metadata["manifest_sha256"] == pins[expected_role + "_manifest_sha256"], "Manifest identity mismatch")
    sources = validate_manifest(manifest)
    require(len(sources) == pins[expected_role + "_examples"] and len(sources) > 0, "Wrong locked panel size")
    require(metadata["example_ids"] == [s["example_id"] for s in sources], "Manifest order/IDs mismatch")
    for key in ("example_id", "image_id", "source_id"):
        require(all(isinstance(s.get(key), str) and s[key] for s in sources) and len({s[key] for s in sources}) == len(sources), f"Repeated/invalid {key}")
    validate_preparation(manifest, metadata, sources, expected_role)
    expected = {(s["example_id"], a) for s in sources for a in ACTIONS}
    records = read_jsonl(run / "records.jsonl")
    require(len(records) == len(expected), "Incomplete or excess record count")
    by_key = {}
    for row in records:
        finite_tree(row, "record")
        key = row.get("example_id"), row.get("action")
        require(key in expected and key not in by_key and row.get("status") == "ok", f"Unexpected, failed or duplicate record: {key}")
        by_key[key] = row
    completion = read_json(run / "completed.json")
    finite_tree(completion, "completion")
    require(completion.get("records") == len(expected), "Completion count mismatch")
    require(completion.get("records_sha256") == sha256_file(run / "records.jsonl"), "Completion record hash mismatch")
    if completion.get("elapsed_s") is not None:
        number(completion["elapsed_s"], "study elapsed")
    else:
        require(completion.get("note") == "Completion marker recovered after all records were durable; total wall time unavailable.", "Missing study elapsed outside documented recovery")
    require(utc(completion["finished_utc"]) >= utc(metadata["created_utc"]), "Completion predates run")
    groups = []
    for source in sources:
        group = {a: by_key[(source["example_id"], a)] for a in ACTIONS}
        image_path = (manifest.parent / source["image_path"]).resolve()
        require(image_path.is_relative_to(manifest.parent) and not Path(source["image_path"]).is_absolute(), "Image escapes manifest directory")
        require(sha256_file(image_path) == source["image_sha256"], "Source image hash mismatch")
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        for action, row in group.items():
            for key, value in (("image_id", source["image_id"]), ("source_id", source["source_id"]),
                               ("question", source["question"]), ("target_answer", source["answer"]),
                               ("source_image_sha256", source["image_sha256"]),
                               ("roi_provenance_sha256", source["roi_provenance_sha256"])):
                require(row.get(key) == value, f"Record/manifest mismatch: {action}/{key}")
            _, images, geometry = build_request(image, source["question"], action, config, source["roi_pixels"])
            for key, value in geometry.items():
                require(type(row.get(key)) is type(value) and row[key] == value, f"Reconstructed provenance mismatch: {source['example_id']}/{action}/{key}")
            grids = [[1, im.height // 16, im.width // 16] for im in images]
            require(row.get("image_grid_thw") == grids, "Processor grid mismatch")
            visual = sum(t * h * w // 4 for t, h, w in grids)
            require(type(row.get("visual_tokens")) is int and row["visual_tokens"] == visual, "Visual token mismatch")
            for key in ("input_tokens", "generated_tokens"):
                require(type(row.get(key)) is int and row[key] > 0, f"Invalid {key}")
            require(row["input_tokens"] > visual and row["generated_tokens"] <= config["answer_max_tokens"], "Invalid token counts")
            require(type(row.get("generation_truncated")) is bool and (not row["generation_truncated"] or row["generated_tokens"] == config["answer_max_tokens"]), "Invalid truncation flag")
            require(row.get("answer_prefix_prefilled") is True and isinstance(row.get("raw_continuation"), str)
                    and row.get("response") == "ANSWER:" + row["raw_continuation"], "Response/prefill mismatch")
            for key in ("elapsed_s", "peak_memory_gib", "peak_reserved_gib"):
                number(row.get(key), key)
            require(row["elapsed_s"] > 0 and row["peak_reserved_gib"] >= row["peak_memory_gib"], "Invalid cost/memory ordering")
            scores = score_response(row["response"], source["answer"])
            scores.update(score_official(scores["predicted_answer"] or "", source["answer"]))
            for key, value in scores.items():
                require(type(row.get(key)) is type(value) and row[key] == value, f"Stored score mismatch: {action}/{key}")
            for metric in METRICS:
                number(row[metric], metric)
                require(row[metric] <= 1, "Metric outside [0,1]")
        for budget in BUDGETS:
            direct, native, degraded = (group[f"{kind}_{budget}"] for kind in ("direct", "native", "degraded"))
            require(direct["image_rgb_sha256"][0] == native["image_rgb_sha256"][0] == degraded["image_rgb_sha256"][0], "Matched overview pixels differ")
            for key in ("messages_sha256", "question_prompt", "overview_size", "additional_size", "source_roi_pixels",
                        "projected_overview_roi_pixels", "crop_box_normalized", "image_grid_thw", "input_tokens", "visual_tokens"):
                require(native[key] == degraded[key], f"Native/degraded pair mismatch: {budget}/{key}")
        natives = [group[f"native_{b}"] for b in BUDGETS]
        for key in ("additional_size", "source_roi_pixels"):
            require(all(r[key] == natives[0][key] for r in natives), f"Native ROI changes across budgets: {key}")
        require(len({r["image_rgb_sha256"][1] for r in natives}) == 1, "Native ROI pixels change across budgets")
        require(len({tuple(r["overview_size"]) for r in natives}) == len(BUDGETS), "Overview budgets do not change overview geometry")
        groups.append(group)
    if "sum_measured_action_elapsed_s" in completion:
        number(completion["sum_measured_action_elapsed_s"], "sum measured action time")
        require(math.isclose(completion["sum_measured_action_elapsed_s"], sum(r["elapsed_s"] for r in records), rel_tol=1e-12, abs_tol=1e-9), "Completion measured-time sum mismatch")
    return metadata, sources, groups, completion


def analyze(run, manifest, config_path=None, *, allow_smoke=False):
    run, manifest = Path(run).resolve(), Path(manifest).resolve()
    config_path = Path(config_path or PROJECT / "configs/evidence_availability.json").resolve()
    role = "smoke" if allow_smoke else "main"
    metadata, sources, groups, completion = validate_run(run, manifest, config_path, expected_role=role)
    result = summarize(groups, metadata["config"])
    example_ids = [s["example_id"] for s in sources]
    for budget, diagnostics in result["paired_native_degraded_response_agreement"].items():
        diagnostics["parsed_answer_changed_example_ids"] = [example_ids[i] for i in diagnostics["parsed_answer_changed_source_indices"]]
        diagnostics["semantic_audit_required_example_ids"] = [example_ids[i] for i in diagnostics["semantic_audit_required_source_indices"]]
    result.update(status="engineering_smoke" if allow_smoke else "complete", experiment="evidence_availability_v5",
        development_only=True, role=role,
        coverage={"source_reports": len(sources), "examples": len(groups), "actions_per_example": len(ACTIONS),
                  "records": len(groups) * len(ACTIONS), "missing_records": 0, "duplicate_records": 0},
        per_source_example_ids=example_ids, model=metadata["model"], runtime=metadata["runtime"], completion=completion,
        integrity={"passed": True, "scores_recomputed": True, "all_image_and_chat_provenance_reconstructed": True,
            "native_degraded_grid_input_tokens_matched": True, "native_roi_pixels_identical_across_budgets": True,
            "source_reports_unique": True, "previous_answer_not_required_or_used": True,
            "preparation_and_annotation_geometry_validated": True,
            "limitation": "No VLM inference or processor forward pass is repeated; processed grids and pixel/chat hashes are reconstructed. ROI selection is annotation privileged."},
        bindings={"run_fingerprint": metadata["fingerprint"], "run_json_sha256": sha256_file(run / "run.json"),
            "records_sha256": sha256_file(run / "records.jsonl"), "completion_sha256": sha256_file(run / "completed.json"),
            "manifest_sha256": sha256_file(manifest), "config_file_sha256": sha256_file(config_path),
            "preparation_metadata_sha256": sha256_file(manifest.parent / "selection_metadata.json"),
            "roi_audit_sha256": sha256_file(manifest.parent / "roi_audit.jsonl"),
            "config_canonical_sha256": canonical_hash(metadata["config"]), "inference_code_sha256": metadata["code_sha256"],
            "base_source_sha256": code_digest(), "analysis_script_sha256": sha256_file(__file__),
            "shared_v3_analysis_helpers_sha256": sha256_file(PROJECT / "experiments/analyze_native_detail.py"),
            "shared_v4_paired_contrast_sha256": sha256_file(PROJECT / "experiments/analyze_history_context.py"),
            "official_metric_upstream_commit": UPSTREAM_COMMIT, "official_metric_upstream_sha256": UPSTREAM_SHA256},
        cost_definitions={"primary": "Standalone synchronized invocation; all conditions use fresh single-turn context.",
            "decision_state": "Optional accounting scenario: matching direct_budget plus chosen native/degraded branch; direct and highres stay standalone. No prior answer is used or required by the branch.",
            "memory": "Sequential peak is maximum, never sum; invocation memory is the actual branch peak.",
            "exclusions": "Download, PDF rendering, model load, warmup and logging excluded; privileged ROI preparation/localization cost is not a deployed selector cost. No deployment saving is established."},
        caveats=["Annotation-guided ROI exposes privileged localization; a positive effect demonstrates conditional evidence availability, not an achieved selection policy.",
            "The panel reuses source reports with known earlier development outcomes; it is not held-out confirmation or transfer.",
            "Official EM is a dataset convention with numeric/sign/rounding quirks. Conservative EM and semantic review remain necessary; F1 is not binary factual correctness.",
            "Only official EM native_256 minus degraded_256 is primary; all other metrics/contrasts are secondary, with unadjusted intervals.",
            "Original labels and preparation eligibility remain fixed. Invalid/truncated generations remain in every denominator.",
            "A quantitative gate pass requires a separate complete semantic audit; it never means controller-ready.",
            "A zero or narrow empirical interval cannot establish population equivalence or rule out rare benefits.",
            "No redundant 94-report sensitivity is computed: source-quality filtering already defines the frozen eligible ROI cohort."])
    if allow_smoke:
        result["candidate_gate"]["applied_to_primary_panel"] = False
        result["candidate_gate"]["quantitative_rule_met"] = False
    else:
        result["candidate_gate"]["applied_to_primary_panel"] = True
    return result


def markdown(result):
    lines = ["# LookAgain v5: evidence availability with privileged ROI", "",
             f"Status: **{result['status']}**. {result['coverage']['source_reports']} source reports; {result['coverage']['records']} complete calls.", "",
             "The region is selected using reference-linked annotation. These are development diagnostics, not learned localization, transfer, or controller performance.", "",
             "## Action means", "", "| Action | Official EM | Conservative EM | Official F1 | ANLS | Invalid / truncated |",
             "|---|---:|---:|---:|---:|---:|"]
    for action, row in result["actions"].items():
        values = " | ".join(f"{row['metrics'][m]*100:.2f}%" for m in METRICS)
        lines.append(f"| {action} | {values} | {row['invalid_answers']} / {row['truncated_generations']} |")
    lines += ["", "## Paired detail effects", "", "The sole primary contrast is native_256 minus degraded_256 on official EM. Intervals resample whole reports; secondary intervals are unadjusted.", "",
              "| Overview budget | Metric | Native minus degraded, pp | 95% interval, pp |", "|---|---|---:|---:|"]
    for budget, metrics in result["detail_effects"].items():
        for metric, stat in metrics.items():
            lo, hi = stat["ci95_pp"]
            lines.append(f"| {budget} | {metric} | {stat['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "## Other paired contrasts", "", "| Contrast | Metric | Difference, pp | 95% interval, pp |", "|---|---|---:|---:|"]
    for name, contrast in result["contrasts"].items():
        if name in (f"native_{b}_minus_degraded_{b}" for b in BUDGETS):
            continue
        for metric, stat in contrast["metrics"].items():
            lo, hi = stat["ci95_pp"]
            lines.append(f"| {name} | {metric} | {stat['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "## Native/degraded transitions", "", "Fixes and harms use degraded as the reference, with explicit eligible denominators.", "",
              "| Budget | Metric | Fixes / degraded-wrong | Harms / degraded-correct | Correct to correct | Wrong to wrong |", "|---|---|---:|---:|---:|---:|"]
    for budget, metrics in result["paired_native_degraded_transitions"].items():
        for metric, row in metrics.items():
            lines.append(f"| {budget} | {metric} | {row['fixes']} / {row['negative_wrong']} | {row['harms']} / {row['negative_correct']} | {row['correct_to_correct']} | {row['wrong_to_wrong']} |")
    lines += ["", "## Costs", "", *[f"- **{key}:** {value}" for key, value in result["cost_definitions"].items()], "",
              "| Action | Standalone seconds mean / median / p95 | Decision-state seconds mean / median / p95 | Standalone peak allocated GiB mean / max |", "|---|---:|---:|---:|"]
    for action, row in result["actions"].items():
        s, d, memory = row["standalone_latency_s"], row["decision_state_latency_s"], row["standalone_peak_allocated_gib"]
        lines.append(f"| {action} | {s['mean']:.3f} / {s['median']:.3f} / {s['p95']:.3f} | {d['mean']:.3f} / {d['median']:.3f} / {d['p95']:.3f} | {memory['mean']:.2f} / {memory['max']:.2f} |")
    gate = result["candidate_gate"]
    changed = result["paired_native_degraded_response_agreement"]["256"]["semantic_audit_required"]
    lines += ["", "## Quantitative screen and semantic audit", "",
              f"Quantitative rule met: **{gate['quantitative_rule_met']}**. Requires primary difference at least {gate['detail_effect_threshold']*100:.1f} pp, interval lower bound above zero, and native_256 accuracy at least direct_256.", "",
              f"Semantic audit: **{gate['semantic_audit_status']}**; {changed} primary pairs require inspection: parsed-answer changes (including equal grades), plus raw differences when either parse is invalid. Controller ready: **False**.", "",
              "## Limitations", "", *[f"- {value}" for value in result["caveats"]], "",
              "## Provenance", "", f"Run fingerprint: `{result['bindings']['run_fingerprint']}`.", "",
              f"Records SHA256: `{result['bindings']['records_sha256']}`.", "",
              f"Manifest SHA256: `{result['bindings']['manifest_sha256']}`.", "",
              "All completed records, source hashes, ROI pixels, chat provenance, paired grids/tokens and grades passed reconstruction. No model inference was repeated.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/evidence_availability.json")
    parser.add_argument("--allow-smoke", action="store_true", help="Validate only the locked engineering-smoke role, never a partial main run")
    args = parser.parse_args()
    result = analyze(args.run, args.manifest, args.config, allow_smoke=args.allow_smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "reports": result["coverage"]["source_reports"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
