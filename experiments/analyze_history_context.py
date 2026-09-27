"""Strict paired report-level analysis of the v4 history/context diagnostic.

No model is loaded. All images, chat provenance and grades are reconstructed.
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
from history_context_core import ACTIONS, HISTORIES, FIDELITIES, REGIONS, build_request, score_response
from history_context import BASE_SOURCE_SHA256, experiment_digest, validate_config, validate_manifest
from native_detail_core import build_request as build_v3_request
from native_detail import experiment_digest as v3_experiment_digest
from lookagain.runner import code_digest, sha256_file
from tatdqa_metrics import score_official, UPSTREAM_COMMIT, UPSTREAM_SHA256
from analyze_native_detail import (canonical_hash, read_json, read_jsonl, require, number,
                                   describe, changes_vs_direct)

METRICS = ("conservative_text_em", "official_em", "official_f1", "anls")
IDENTITY_KEYS = ("config", "manifest_sha256", "code_sha256", "model", "runtime", "role", "example_ids")
CONDITIONS = {"direct": ["direct"], "highres": ["highres"], "repeat": ["repeat"],
              **{f"{h}_{f}": [f"{h}_{f}_{r}" for r in REGIONS]
                 for h in HISTORIES for f in FIDELITIES}}
PRIMARY = "placeholder_minus_actual_detail_interaction"
V3_PINS = {"run.json": "6e0b7ab5c5851457b575260eef0e8eb7c03ac7b262f406de8c308f1db9bc8f7a",
           "records.jsonl": "f55389457f0bf109326686ab629ffd351be6290d592242cc9e8e7c346d1e2805",
           "completed.json": "d054c3d3aef9fd70501b8fc860451fdd6c809d987544633169d2854c6cb216e9"}
V3_CODE = "515dc6941bfe514a3beffba157d8b31a54ae061525447a1653c1dc57bcda57cd"


def utc(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.tzinfo is not None, "Timestamp must include a timezone")
    return stamp


def normalize_answer(value):
    return " ".join(value.casefold().split()) if isinstance(value, str) else None


def paired_contrast(positive, negative, draws):
    """Shared source-index draws preserve quadrant and history pairing."""
    require(len(positive) == len(negative) and len(positive) > 0, "Unpaired or empty contrast")
    delta = np.asarray(positive, dtype=float) - np.asarray(negative, dtype=float)
    require(bool(np.isfinite(delta).all()), "Nonfinite contrast")
    samples = delta[np.asarray(draws, dtype=np.int32)].mean(axis=1)
    ci = np.quantile(samples, [.025, .975]).tolist()
    point = float(delta.mean())
    return {"n_source_reports": len(delta), "difference": point, "difference_pp": point * 100,
            "ci95": ci, "ci95_pp": [x * 100 for x in ci]}


def cost(group, action, field="elapsed_s", mode="decision_state"):
    require(mode in ("decision_state", "standalone"), "Unknown cost perspective")
    row = group[action]
    needs_direct = action not in ("direct", "highres") and (
        mode == "decision_state" or action == "repeat" or action.startswith("actual_"))
    if not needs_direct:
        return row[field]
    if field in ("peak_memory_gib", "peak_reserved_gib"):
        return max(row[field], group["direct"][field])
    return row[field] + group["direct"][field]


def summarize_actions(groups, actions):
    pairs = [(g, a) for g in groups for a in actions]
    rows = [g[a] for g, a in pairs]
    result = {"n_source_reports": len(groups), "n_presentations": len(rows),
              "metrics": {m: statistics.fmean(r[m] for r in rows) for m in METRICS},
              "invalid_answers": sum(not r["parse_valid"] for r in rows),
              "truncated_generations": sum(r["generation_truncated"] for r in rows),
              "invocation_latency_s": describe([r["elapsed_s"] for r in rows])}
    for mode in ("decision_state", "standalone"):
        result[mode + "_latency_s"] = describe([cost(g, a, mode=mode) for g, a in pairs])
        result[mode + "_peak_allocated_gib"] = describe([cost(g, a, "peak_memory_gib", mode) for g, a in pairs])
        result[mode + "_peak_reserved_gib"] = describe([cost(g, a, "peak_reserved_gib", mode) for g, a in pairs])
    result["tokens"] = {field: {"invocation": describe([r[field] for r in rows]),
                               **{mode: describe([cost(g, a, field, mode) for g, a in pairs])
                                  for mode in ("decision_state", "standalone")}}
                        for field in ("input_tokens", "visual_tokens", "generated_tokens")}
    return result


def response_persistence(groups, actions):
    pairs = [(g["direct"], g[a]) for g in groups for a in actions]
    valid = [(d, r) for d, r in pairs if d["parse_valid"] and r["parse_valid"]]
    counts = {
        "raw_response_exact": sum(d["response"] == r["response"] for d, r in pairs),
        "parsed_answer_exact": sum(d["predicted_answer"] == r["predicted_answer"] for d, r in valid),
        "normalized_answer": sum(normalize_answer(d["predicted_answer"]) == normalize_answer(r["predicted_answer"])
                                 for d, r in valid)}
    return {"n_presentations": len(pairs), "valid_direct_and_branch": len(valid),
            "invalid_pairs_retained_in_denominator": len(pairs) - len(valid),
            "counts": counts, "fractions": {k: v / len(pairs) for k, v in counts.items()},
            "normalization": "Parsed answer, casefold and whitespace collapse only; punctuation and units retained.",
            "count_scope": "Raw equality includes all strings, including invalid parses; parsed/normalized equality requires both parses valid. Every fraction uses all presentations.",
            "interpretation": "Descriptive agreement with the current direct answer, not proof of an internal mechanism."}


def paired_response_agreement(groups, left_actions, right_actions):
    require(len(left_actions) == len(right_actions), "Unpaired action sets")
    pairs = [(g[a], g[b]) for g in groups for a, b in zip(left_actions, right_actions)]
    valid = [(a, b) for a, b in pairs if a["parse_valid"] and b["parse_valid"]]
    return {"n_presentations": len(pairs), "valid_pairs": len(valid),
            "raw_response_exact": sum(a["response"] == b["response"] for a, b in pairs),
            "normalized_answer_equal": sum(normalize_answer(a["predicted_answer"]) == normalize_answer(b["predicted_answer"])
                                           for a, b in valid),
            "metric_equal": {m: sum(a[m] == b[m] for a, b in pairs) for m in METRICS}}


def summarize(groups, config, bootstrap=None):
    require(bool(groups), "No complete source groups")
    analysis = config["analysis"]
    require(analysis["primary_metric"] == "conservative_text_em", "Unexpected primary metric")
    samples = analysis.get("bootstrap_samples", 10000) if bootstrap is None else bootstrap
    seed = analysis.get("bootstrap_seed", 20260925)
    require(type(samples) is int and samples > 0 and type(seed) is int, "Invalid bootstrap configuration")
    n = len(groups)
    rng = random.Random(seed)
    draws = np.asarray([[rng.randrange(n) for _ in range(n)] for _ in range(samples)], dtype=np.int32)
    vectors = {m: {name: [statistics.fmean(g[a][m] for a in actions) for g in groups]
                   for name, actions in CONDITIONS.items()} for m in METRICS}
    detail = {m: {h: [a - b for a, b in zip(vectors[m][h + "_native"], vectors[m][h + "_degraded"])]
                  for h in HISTORIES} for m in METRICS}
    contrasts = {}
    for h in HISTORIES:
        for f in FIDELITIES:
            positive = h + "_" + f
            for other in ([h + "_degraded"] if f == "native" else []) + ["direct", "highres", "repeat"]:
                name = positive + "_minus_" + other
                contrasts[name] = {"role": "secondary", "positive": positive, "negative": other,
                                   "metrics": {m: paired_contrast(vectors[m][positive], vectors[m][other], draws) for m in METRICS}}
    for h, reference in (("placeholder", "actual"), ("fresh", "actual"), ("fresh", "placeholder")):
        name = h + "_minus_" + reference + "_detail_interaction"
        contrasts[name] = {"role": "primary interaction; conservative EM only" if name == PRIMARY else "secondary interaction",
                           "definition": f"({h}_native - {h}_degraded) - ({reference}_native - {reference}_degraded), within each source report",
                           "metrics": {m: paired_contrast(detail[m][h], detail[m][reference], draws) for m in METRICS}}
    for positive in ("highres", "repeat"):
        contrasts[positive + "_minus_direct"] = {"role": "secondary", "positive": positive, "negative": "direct",
            "metrics": {m: paired_contrast(vectors[m][positive], vectors[m]["direct"], draws) for m in METRICS}}
    oracles = {}
    oracle_vectors = {m: {} for m in METRICS}
    for h in HISTORIES:
        for f in FIDELITIES:
            name = f"{h}_{f}"
            candidates = ["direct", *CONDITIONS[name]]
            for m in METRICS:
                oracle_vectors[m][name] = [max(g[a][m] for a in candidates) for g in groups]
            selected = [min(candidates, key=lambda a: (-g[a]["conservative_text_em"], cost(g, a), ACTIONS.index(a))) for g in groups]
            oracles[name] = {"candidate_actions": candidates,
                "metrics": {m: statistics.fmean(oracle_vectors[m][name]) for m in METRICS},
                "primary_selected_action_counts": {a: selected.count(a) for a in candidates},
                "optimistic_selected_decision_state_latency_s": describe([cost(g, a) for g, a in zip(groups, selected)]),
                "gain_beyond_highres": {m: statistics.fmean(oracle_vectors[m][name]) - statistics.fmean(vectors[m]["highres"]) for m in METRICS}}
    detail_effects = {h: {m: paired_contrast(detail[m][h], [0.] * n, draws) for m in METRICS} for h in HISTORIES}
    threshold = analysis.get("candidate_gate", {}).get("detail_effect_threshold", .02)
    gate_histories = {}
    for h in HISTORIES:
        effect = detail_effects[h]["conservative_text_em"]
        checks = {"detail_mean_at_least_threshold": effect["difference"] >= threshold,
                  "detail_ci95_lower_positive": effect["ci95"][0] > 0,
                  "native_mean_at_least_direct": statistics.fmean(vectors["conservative_text_em"][h + "_native"]) >= statistics.fmean(vectors["conservative_text_em"]["direct"])}
        gate_histories[h] = {"checks": checks, "candidate_rule_met": all(checks.values())}
    return {"conditions": {name: summarize_actions(groups, actions) for name, actions in CONDITIONS.items()},
            "actions": {a: summarize_actions(groups, [a]) for a in ACTIONS}, "contrasts": contrasts,
            "detail_effects": detail_effects,
            "candidate_gate": {"metric": "conservative_text_em", "detail_effect_threshold": threshold,
                "histories": gate_histories, "any_candidate_rule_met": any(x["candidate_rule_met"] for x in gate_histories.values()),
                "automatic_controller_training": False, "primary_interaction_remains_primary": True,
                "interpretation": "Exploratory unadjusted screen for validation on untouched original reports; not a training authorization or held-out result."},
            "per_source_condition_scores": {m: vectors[m] for m in METRICS},
            "changes_vs_direct": {name: changes_vs_direct(groups, actions) for name, actions in CONDITIONS.items()},
            "action_changes_vs_direct": {a: changes_vs_direct(groups, [a]) for a in ACTIONS},
            "response_persistence": {name: response_persistence(groups, actions) for name, actions in CONDITIONS.items()},
            "paired_native_degraded_response_agreement": {h: paired_response_agreement(groups, CONDITIONS[h + "_native"], CONDITIONS[h + "_degraded"]) for h in HISTORIES},
            "cross_history_response_agreement": {f"{left}_vs_{right}_{f}": paired_response_agreement(groups, CONDITIONS[left + "_" + f], CONDITIONS[right + "_" + f])
                for left, right in (("actual", "fresh"), ("actual", "placeholder"), ("fresh", "placeholder")) for f in FIDELITIES},
            "privileged_oracles": {"label": "Five-candidate reference-aware upper bounds; not policies. Each metric selects separately.",
                "cost_warning": "Selected costs omit selector/search costs and use reference labels; optimistic, not achieved performance.",
                "oracles": oracles,
                "matched_native_minus_degraded": {h: {m: paired_contrast(oracle_vectors[m][h + "_native"], oracle_vectors[m][h + "_degraded"], draws) for m in METRICS} for h in HISTORIES}},
            "bootstrap": {"samples": samples, "seed": seed, "unit": "original_source_report", "n_source_reports": n,
                "retains_all_histories_fidelities_and_quadrants": True, "interval": "95% percentile; exploratory/unadjusted",
                "resampling": "Shared random.Random(seed).randrange(n) source draws; interactions formed within source before resampling."}}


def validate_run(run, manifest, config_path, *, expected_role="main"):
    run, manifest, config_path = Path(run).resolve(), Path(manifest).resolve(), Path(config_path).resolve()
    metadata = read_json(run / "run.json")
    require(canonical_hash({k: metadata[k] for k in IDENTITY_KEYS}) == metadata.get("fingerprint"), "Run fingerprint mismatch")
    config = metadata["config"]
    require(config == read_json(config_path), "Run config differs from supplied locked config")
    validate_config(config)
    require(metadata.get("experiment") == "history_context_v4" and metadata.get("development_only") is True, "Unexpected experiment")
    require(expected_role in ("main", "smoke") and metadata.get("role") == expected_role, "Only the requested complete locked panel can be validated")
    require(metadata.get("base_code_sha256") == code_digest() == BASE_SOURCE_SHA256, "Frozen base source differs")
    require(metadata["code_sha256"] == experiment_digest() == config["protocol"]["inference_code_sha256"], "V4 inference code differs")
    require(v3_experiment_digest() == V3_CODE, "Frozen v3 inference files differ")
    pins = config["protocol"]
    require(metadata["model"] == {"model_id": config["model_id"], "revision": pins["model_revision"]}, "Model identity mismatch")
    for key in ("rules_locked_at_utc", "main_execution_locked_at_utc"):
        require(utc(pins[key]) <= utc(metadata["created_utc"]), f"Run predates {key}")
    for k, value in metadata["runtime"].items():
        require(metadata.get(k) == value, f"Inconsistent runtime field {k}")
    require(sha256_file(manifest) == metadata["manifest_sha256"] == pins[expected_role + "_manifest_sha256"], "Manifest identity mismatch")
    sources = validate_manifest(manifest)
    require(len(sources) == pins[expected_role + "_examples"] and len(sources) > 0, "Wrong locked panel size")
    require(metadata["example_ids"] == [r["example_id"] for r in sources], "Manifest order/IDs mismatch")
    expected = {(r["example_id"], a) for r in sources for a in ACTIONS}
    records = read_jsonl(run / "records.jsonl")
    require(len(records) == len(expected), "Incomplete or excess record count")
    by_key = {}
    for row in records:
        key = row.get("example_id"), row.get("action")
        require(key in expected and key not in by_key and row.get("status") == "ok", f"Unexpected, failed or duplicate record: {key}")
        by_key[key] = row
    completion = read_json(run / "completed.json")
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
        with Image.open(manifest.parent / source["image_path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        initial = group["direct"]["response"]
        for action, row in group.items():
            require(row.get("observed_direct_response_sha256") == hashlib.sha256(initial.encode()).hexdigest(), "Observed direct response hash mismatch")
            for k, value in (("image_id", source["image_id"]), ("source_id", source["source_id"]),
                             ("question", source["question"]), ("target_answer", source["answer"]),
                             ("source_image_sha256", source["image_sha256"])):
                require(row.get(k) == value, f"Record/manifest mismatch: {action}/{k}")
            messages, images, geometry = build_request(image, source["question"], action, config, initial)
            for k, value in geometry.items():
                require(type(row.get(k)) is type(value) and row[k] == value, f"Reconstructed provenance mismatch: {source['example_id']}/{action}/{k}")
            grids = [[1, im.height // 16, im.width // 16] for im in images]
            require(row.get("image_grid_thw") == grids, "Processor grid mismatch")
            visual = sum(t * h * w // 4 for t, h, w in grids)
            require(type(row.get("visual_tokens")) is int and row["visual_tokens"] == visual, "Visual token mismatch")
            for k in ("input_tokens", "generated_tokens"):
                require(type(row.get(k)) is int and row[k] > 0, f"Invalid {k}")
            require(row["input_tokens"] > visual and row["generated_tokens"] <= config["answer_max_tokens"], "Invalid token counts")
            require(type(row.get("generation_truncated")) is bool and (not row["generation_truncated"] or row["generated_tokens"] == config["answer_max_tokens"]), "Invalid truncation flag")
            require(row.get("answer_prefix_prefilled") is True and isinstance(row.get("raw_continuation"), str)
                    and row["response"] == "ANSWER:" + row["raw_continuation"], "Response/prefill mismatch")
            for k in ("elapsed_s", "peak_memory_gib", "peak_reserved_gib"):
                number(row.get(k), k)
            require(row["elapsed_s"] > 0 and row["peak_reserved_gib"] >= row["peak_memory_gib"], "Invalid cost/memory ordering")
            scores = score_response(row["response"], source["answer"])
            scores.update(score_official(scores["predicted_answer"] or "", source["answer"]))
            for k, value in scores.items():
                require(type(row.get(k)) is type(value) and row[k] == value, f"Stored score mismatch: {action}/{k}")
            for m in METRICS:
                number(row[m], m)
                require(row[m] <= 1, "Metric outside [0,1]")
            if action in ("direct", "highres", "repeat") or action.startswith("actual_"):
                old_action = action.removeprefix("actual_")
                old_messages, _, old_geometry = build_v3_request(image, source["question"], old_action, config, initial)
                require(messages == old_messages, "Actual-history chat differs from v3 construction")
                require(geometry["image_rgb_sha256"] == old_geometry["image_rgb_sha256"], "Actual-history pixels differ from v3 construction")
        direct_hash = group["direct"]["image_rgb_sha256"][0]
        for action in ACTIONS:
            if action not in ("direct", "highres"):
                require(group[action]["image_rgb_sha256"][0] == direct_hash, "First overview changed")
        require(group["repeat"]["image_rgb_sha256"] == [direct_hash, direct_hash], "Repeat is not byte-identical RGB")
        for region in REGIONS:
            for history in HISTORIES:
                left, right = (group[f"{history}_{f}_{region}"] for f in FIDELITIES)
                for k in ("messages_sha256", "initial_answer_sha256", "inserted_history_text", "previous_answer_in_prompt",
                          "overview_size", "additional_size", "source_roi_pixels", "projected_overview_roi_pixels",
                          "image_grid_thw", "input_tokens", "visual_tokens", "followup_prompt"):
                    require(left[k] == right[k], f"Native/degraded pair mismatch: {history}/{region}/{k}")
            for fidelity in FIDELITIES:
                matched = [group[f"{h}_{fidelity}_{region}"] for h in HISTORIES]
                for k in ("image_rgb_sha256", "image_grid_thw", "visual_tokens", "source_roi_pixels", "additional_size"):
                    require(all(r[k] == matched[0][k] for r in matched), f"Cross-history pixels/geometry mismatch: {fidelity}/{region}/{k}")
        groups.append(group)
    return metadata, sources, groups, completion


def validate_quality_mask(path, metadata, sources):
    require(path is not None, "Supply the frozen --quality-mask")
    pinned = metadata["config"]["analysis"]["quality_mask_sha256"]
    require(sha256_file(path) == pinned, "Quality-mask hash mismatch")
    mask = read_json(path)
    require(mask.get("frozen_before_main_inference") is True and mask.get("review_is_before_main_inference") is True
            and mask.get("model_predictions_or_outcomes_accessed_by_reviewers") is False, "Mask is not source-only and pre-inference")
    require(mask["main_manifest_sha256"] == metadata["manifest_sha256"], "Mask manifest mismatch")
    for key in ("locked_at_utc", "created_utc", "frozen_at_utc"):
        require(utc(mask[key]) <= utc(metadata["created_utc"]), "Quality mask postdates inference")
    main = {r["example_id"]: r for r in sources}
    cats, details = mask["categories_by_example_id"], mask["mask"]
    require(set(main) <= set(cats) and set(main) <= set(details), "Mask misses main IDs")
    excluded = mask["secondary_excluded_main_ids"]
    require(len(excluded) == len(set(excluded)) and set(excluded) <= set(main), "Invalid exclusion list")
    require(set(excluded) == {k for k in main if cats[k] in ("label_error", "ambiguous")}, "Mask category/exclusion mismatch")
    for key, source in main.items():
        d = details[key]
        require(cats[key] in ("none", "mapping_only", "label_error", "ambiguous") and d["category"] == cats[key] and d["cohort"] == "main", "Invalid mask category")
        for a, b in (("source_id", "source_id"), ("source_image_sha256", "image_sha256"), ("question", "question"), ("reference_answer", "answer")):
            require(d[a] == source[b], f"Mask source binding mismatch: {key}/{a}")
        if key in excluded:
            require(d.get("full_page_visual_check") is True and bool(d.get("reason", "").strip()), "Excluded source lacks visual verification/reason")
    return mask, set(excluded)


def archived_v3_agreement(path, metadata, groups):
    path = Path(path).resolve()
    pins = metadata["config"]["protocol"]
    names = {"run.json": "v3_run_sha256", "records.jsonl": "v3_records_sha256", "completed.json": "v3_completion_sha256"}
    for filename, pin_key in names.items():
        expected = pins.get(pin_key, V3_PINS[filename])
        require(expected == V3_PINS[filename] and sha256_file(path / filename) == expected, f"Archived v3 binding mismatch: {filename}")
    old = read_json(path / "run.json")
    require(canonical_hash({k: old[k] for k in IDENTITY_KEYS}) == old["fingerprint"], "Archived v3 fingerprint mismatch")
    require(old["manifest_sha256"] == metadata["manifest_sha256"] and old["model"] == metadata["model"], "V3/v4 model or cohort differs")
    require(old["runtime"] == metadata["runtime"], "V3/v4 runtime identities differ")
    previous = {(r["example_id"], r["action"]): r for r in read_jsonl(path / "records.jsonl")}
    mappings = {a: a.removeprefix("actual_") for a in ACTIONS if a in ("direct", "highres", "repeat") or a.startswith("actual_")}
    output = {}
    for action, old_action in mappings.items():
        pairs = [(g[action], previous[(g["direct"]["example_id"], old_action)]) for g in groups]
        require(all(a["image_rgb_sha256"] == b["image_rgb_sha256"] for a, b in pairs), "V4 actual branch pixels differ from archived v3")
        exact = sum(a["response"] == b["response"] for a, b in pairs)
        normalized = sum(a["parse_valid"] and b["parse_valid"] and normalize_answer(a["predicted_answer"]) == normalize_answer(b["predicted_answer"]) for a, b in pairs)
        output[action] = {"n": len(pairs), "exact_response_equal": exact, "normalized_answer_equal": normalized,
            "chat_hash_equal": sum(a["messages_sha256"] == b["messages_sha256"] for a, b in pairs),
            "metric_mean_differences": {m: statistics.fmean(a[m] - b[m] for a, b in pairs) for m in METRICS}}
    return {"role": "Descriptive same-data repeat agreement only; no independent replication claim and no timing inference.",
            "actual_chat_equivalence": "Validated against v3 construction using the same current direct response; archived chat hashes can differ if that response differs.",
            "run_fingerprint": old["fingerprint"], "hashes": V3_PINS, "actions": output}


def analyze(run, manifest, quality_mask, config_path=None, bootstrap=None, v3_run=None):
    run, manifest = Path(run).resolve(), Path(manifest).resolve()
    config_path = Path(config_path or PROJECT / "configs/history_context.json").resolve()
    metadata, sources, groups, completion = validate_run(run, manifest, config_path)
    locked_samples = metadata["config"]["analysis"]["bootstrap_samples"]
    require(bootstrap is None or bootstrap == locked_samples, "Bootstrap override differs from locked analysis")
    result = summarize(groups, metadata["config"], bootstrap)
    mask, excluded = validate_quality_mask(quality_mask, metadata, sources)
    retained = [g for g in groups if g["direct"]["example_id"] not in excluded]
    require(bool(retained), "Quality mask excludes every main report")
    require(len(retained) == metadata["config"]["analysis"]["expected_secondary_sources"], "Frozen sensitivity size mismatch")
    sensitivity = summarize(retained, metadata["config"], bootstrap)
    sensitivity.update(role="Secondary frozen source-quality sensitivity; all-report primary unchanged",
        retained_n_source_reports=len(retained), excluded_example_ids=sorted(excluded),
        primary_n_source_reports=len(groups), mask_sha256=sha256_file(quality_mask),
        per_source_example_ids=[g["direct"]["example_id"] for g in retained])
    sensitivity.pop("candidate_gate")
    sensitivity["candidate_gate_applied"] = False
    result.update(status="complete", experiment="history_context_v4", development_only=True,
        coverage={"source_reports": len(sources), "examples": len(groups), "actions_per_example": len(ACTIONS), "records": len(groups) * len(ACTIONS), "missing_records": 0, "duplicate_records": 0},
        per_source_example_ids=[s["example_id"] for s in sources], source_quality_sensitivity=sensitivity,
        primary_contrast=PRIMARY, primary_metric="conservative_text_em", model=metadata["model"], runtime=metadata["runtime"], completion=completion,
        integrity={"passed": True, "scores_recomputed": True, "all_image_and_chat_provenance_reconstructed": True,
            "native_degraded_grid_input_tokens_matched": True, "cross_history_pixels_identical": True,
            "actual_history_matches_v3_request_construction": True,
            "limitation": "No VLM inference or processor forward pass is repeated; grids and pixel/chat hashes are reconstructed. Full raw annotation/PDF eligibility remains bound through manifests/preparation audit."},
        bindings={"run_fingerprint": metadata["fingerprint"], "run_json_sha256": sha256_file(run / "run.json"),
            "records_sha256": sha256_file(run / "records.jsonl"), "completion_sha256": sha256_file(run / "completed.json"),
            "manifest_sha256": sha256_file(manifest), "config_file_sha256": sha256_file(config_path),
            "config_canonical_sha256": canonical_hash(metadata["config"]), "inference_code_sha256": metadata["code_sha256"],
            "v3_inference_code_sha256": v3_experiment_digest(), "base_source_sha256": code_digest(),
            "source_quality_mask_sha256": sha256_file(quality_mask), "analysis_script_sha256": sha256_file(__file__),
            "shared_v3_analysis_helpers_sha256": sha256_file(PROJECT / "experiments/analyze_native_detail.py"),
            "official_metric_upstream_commit": UPSTREAM_COMMIT, "official_metric_upstream_sha256": UPSTREAM_SHA256},
        cost_definitions={"decision_state": "Direct plus chosen branch for actual/fresh/placeholder/repeat; direct and highres are standalone comparator calls.",
            "standalone": "Fresh/placeholder invocation alone. Actual/repeat require direct plus branch. Sequential memory is the maximum, never the sum.",
            "exclusions": "Download, PDF rendering, model load, warmup and logging excluded; no selector/proposal is implemented. All timing observations retained."},
        caveats=["Same 100 reports have known v3 outcomes and motivated this diagnostic; not held-out confirmation or transfer.",
            "Only the placeholder-minus-actual detail interaction on conservative EM is primary; all other metrics/contrasts are secondary and intervals unadjusted.",
            "Four-region means describe uniform single-region choice in expectation; no region is chosen by a learned policy.",
            "Fresh also changes conversation structure and wording; placeholder preserves the turn template but changes content and token length.",
            "A history interaction or answer persistence cannot identify an internal anchoring mechanism by itself.",
            "A collapsed or narrow empirical interval does not establish population equivalence, absence of rare benefits, or a mechanistic zero.",
            "Invalid and truncated responses remain in all denominators; original labels and the v3 source-quality mask remain unchanged.",
            "No v3 stopping-rule or controller-training claim is transferred to this new diagnostic."])
    result["archived_v3_agreement"] = archived_v3_agreement(v3_run or PROJECT / "reports/native100", metadata, groups)
    return result


def markdown(result):
    lines = ["# LookAgain v4: history and visual-detail diagnostic", "",
        f"Complete development panel: **{result['coverage']['source_reports']} reports**, **{result['coverage']['records']} calls**. Strict provenance and grade checks passed; no inference was repeated.", "",
        "This reuses reports whose v3 outcomes informed the study. It is a development diagnostic, not held-out confirmation.", "",
        "## Condition means", "", "Four quadrants are averaged within each report; these are not four independent observations.", "",
        "| Condition | Conservative EM | Official EM | Official F1 | ANLS | Invalid / truncated |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in result["conditions"].items():
        values = " | ".join(f"{row['metrics'][m]*100:.2f}%" for m in METRICS)
        lines.append(f"| {name} | {values} | {row['invalid_answers']} / {row['truncated_generations']} |")
    lines += ["", "## Paired source-level contrasts", "",
        "The sole primary estimand is (placeholder native − degraded) − (actual native − degraded), on conservative EM. All intervals are exploratory 95% percentile intervals without multiplicity adjustment.", "",
        "| Contrast | Metric | Difference, pp | 95% interval, pp |", "|---|---|---:|---:|"]
    ordered = [PRIMARY, "fresh_minus_actual_detail_interaction", "fresh_minus_placeholder_detail_interaction"] + [k for k in result["contrasts"] if "detail_interaction" not in k]
    for name in ordered:
        for metric, stat in result["contrasts"][name]["metrics"].items():
            lo, hi = stat["ci95_pp"]
            lines.append(f"| {name} | {metric} | {stat['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "## Repairs, harms and response persistence", "",
        "Raw persistence compares all strings, including invalid parses. Parsed/normalized agreement requires both parses valid; all pairs remain in denominators. It is descriptive, not mechanistic evidence.", "",
        "| Condition | Fixes / direct-wrong | Harms / direct-correct | Exact response persistence | Normalized answer persistence |", "|---|---:|---:|---:|---:|"]
    for name, row in result["changes_vs_direct"].items():
        per = result["response_persistence"][name]
        lines.append(f"| {name} | {row['fixes']} / {row['direct_wrong_presentations']} | {row['harms']} / {row['direct_correct_presentations']} | {per['fractions']['raw_response_exact']*100:.2f}% | {per['fractions']['normalized_answer']*100:.2f}% |")
    lines += ["", "## Cost perspectives", "", *[f"- **{k}:** {v}" for k, v in result["cost_definitions"].items()], "",
        "| Action | Decision-state seconds mean / median / p95 | Standalone seconds mean / median / p95 | Decision-state peak allocated GiB mean / max |", "|---|---:|---:|---:|"]
    for name, row in result["actions"].items():
        d, s, m = row["decision_state_latency_s"], row["standalone_latency_s"], row["decision_state_peak_allocated_gib"]
        lines.append(f"| {name} | {d['mean']:.3f} / {d['median']:.3f} / {d['p95']:.3f} | {s['mean']:.3f} / {s['median']:.3f} / {s['p95']:.3f} | {m['mean']:.2f} / {m['max']:.2f} |")
    lines += ["", "## Matched privileged oracles", "", "Each set has five actions: direct plus four fixed regions. Labels choose the best response; these are not achieved policies or controller costs.", "",
        "| Candidate set | Conservative EM | Official EM | Official F1 | ANLS |", "|---|---:|---:|---:|---:|"]
    for name, row in result["privileged_oracles"]["oracles"].items():
        lines.append("| " + name + " | " + " | ".join(f"{row['metrics'][m]*100:.2f}%" for m in METRICS) + " |")
    sen = result["source_quality_sensitivity"]
    lines += ["", "## Frozen source-quality sensitivity", "",
        f"Retained {sen['retained_n_source_reports']} of {sen['primary_n_source_reports']} reports. This secondary subset does not replace the all-report primary analysis; source flags and reference answers are unchanged.", "",
        "| Contrast | Conservative-EM difference, pp | 95% interval, pp |", "|---|---:|---:|"]
    for name in [PRIMARY, "fresh_minus_actual_detail_interaction", "fresh_minus_placeholder_detail_interaction"]:
        stat = sen["contrasts"][name]["metrics"]["conservative_text_em"]
        lo, hi = stat["ci95_pp"]
        lines.append(f"| {name} | {stat['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "## Candidate screen", "", result["candidate_gate"]["interpretation"], "",
        "The primary interaction remains primary. Each candidate requires detail gain ≥2 percentage points, a positive lower confidence bound, and native quality at least direct. These checks do not trigger controller training.", "",
        "| History | Gain threshold | Lower bound positive | Native ≥ direct | All checks |", "|---|---|---|---|---|"]
    for history, row in result["candidate_gate"]["histories"].items():
        checks = row["checks"]
        lines.append(f"| {history} | {checks['detail_mean_at_least_threshold']} | {checks['detail_ci95_lower_positive']} | {checks['native_mean_at_least_direct']} | {row['candidate_rule_met']} |")
    lines += ["", "## Archived v3 agreement", "", result["archived_v3_agreement"]["role"], "",
        "| Action | Exact response matches | Normalized answer matches | Chat hash matches |", "|---|---:|---:|---:|"]
    for name, row in result["archived_v3_agreement"]["actions"].items():
        lines.append(f"| {name} | {row['exact_response_equal']} / {row['n']} | {row['normalized_answer_equal']} / {row['n']} | {row['chat_hash_equal']} / {row['n']} |")
    lines += ["", "## Limits", "", *["- " + x for x in result["caveats"]], "", "## Bindings", ""]
    lines += [f"- {k}: `{v}`" for k, v in result["bindings"].items() if isinstance(v, str)]
    lines += ["", "The JSON includes all action/condition metrics, token and memory distributions, per-source condition vectors, paired contrasts, persistence, repairs/harms and all source-quality sensitivity results.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--quality-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/history_context.json")
    parser.add_argument("--v3-run", type=Path, default=PROJECT / "reports/native100")
    parser.add_argument("--bootstrap", type=int, default=None, help="Defaults to locked count (10000); override must equal the lock")
    args = parser.parse_args()
    result = analyze(args.run, args.manifest, args.quality_mask, args.config, args.bootstrap, args.v3_run)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(markdown(result), encoding="utf-8")
    print(f"Wrote strict completed-panel v4 analysis to {args.output}")


if __name__ == "__main__":
    main()
