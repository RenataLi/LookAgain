"""Strict completed-panel analysis of the v3 document-detail intervention."""
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

from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT / "src"))
from native_detail_core import ACTIONS, REGIONS, ANSWER_INSTRUCTION, CROP_PROMPT, REPEAT_PROMPT, build_request, score_response
from native_detail import BASE_SOURCE_SHA256, experiment_digest, validate_config, validate_manifest
from lookagain.runner import code_digest, sha256_file
from tatdqa_metrics import score_official, UPSTREAM_COMMIT, UPSTREAM_SHA256

METRICS = ("conservative_text_em", "official_em", "official_f1", "anls")
IDENTITY_KEYS = ("config", "manifest_sha256", "code_sha256", "model", "runtime", "role", "example_ids")
CONDITIONS = {"direct": ["direct"], "highres": ["highres"], "repeat": ["repeat"],
              "native": ["native_" + r for r in REGIONS], "degraded": ["degraded_" + r for r in REGIONS]}
CONTRASTS = (("native_minus_degraded", "native", "degraded"),
             ("native_minus_direct", "native", "direct"),
             ("native_minus_repeat", "native", "repeat"),
             ("degraded_minus_repeat", "degraded", "repeat"),
             ("highres_minus_direct", "highres", "direct"),
             ("native_minus_highres", "native", "highres"))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f"Invalid numeric {label}")
    return value


def quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def describe(values):
    require(bool(values), "Cannot summarize an empty collection")
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": quantile(values, .95), "min": min(values), "max": max(values)}


def paired_contrast(positive, negative, draws):
    require(len(positive) == len(negative) and len(positive) > 0, "Unpaired or empty contrast")
    delta = [a - b for a, b in zip(positive, negative)]
    samples = [statistics.fmean(delta[index] for index in indices) for indices in draws]
    interval = [quantile(samples, .025), quantile(samples, .975)]
    point = statistics.fmean(delta)
    return {"n_source_reports": len(delta), "difference": point, "difference_pp": point * 100,
            "ci95": interval, "ci95_pp": [value * 100 for value in interval]}


def full_cost(group, action, field="elapsed_s"):
    row = group[action]
    if action in ("direct", "highres"):
        return row[field]
    if field in ("peak_memory_gib", "peak_reserved_gib"):
        return max(row[field], group["direct"][field])
    return row[field] + group["direct"][field]


def summarize_actions(groups, actions):
    pairs = [(group, action) for group in groups for action in actions]
    rows = [group[action] for group, action in pairs]
    return {
        "n_source_reports": len(groups), "n_presentations": len(rows),
        "metrics": {metric: statistics.fmean(row[metric] for row in rows) for metric in METRICS},
        "invalid_answers": sum(not row["parse_valid"] for row in rows),
        "truncated_generations": sum(row["generation_truncated"] for row in rows),
        "full_path_latency_s": describe([full_cost(group, action) for group, action in pairs]),
        "sequential_peak_allocated_gib": describe([full_cost(group, action, "peak_memory_gib") for group, action in pairs]),
        "sequential_peak_reserved_gib": describe([full_cost(group, action, "peak_reserved_gib") for group, action in pairs]),
        "tokens": {name: {
            "invocation": describe([row[name] for row in rows]),
            "full_policy": describe([full_cost(group, action, name) for group, action in pairs]),
        } for name in ("input_tokens", "visual_tokens", "generated_tokens")},
    }


def changes_vs_direct(groups, actions):
    pairs = [(group, action) for group in groups for action in actions]
    fixes = sum(not group["direct"]["correct"] and group[action]["correct"] for group, action in pairs)
    harms = sum(group["direct"]["correct"] and not group[action]["correct"] for group, action in pairs)
    wrong = sum(not group["direct"]["correct"] for group, _ in pairs)
    right = len(pairs) - wrong
    gains = [group[action]["anls"] - group["direct"]["anls"] for group, action in pairs]
    positive, negative = [v for v in gains if v > 0], [v for v in gains if v < 0]
    return {
        "n_presentations": len(pairs), "fixes": fixes, "harms": harms,
        "net_correct_presentations": fixes - harms,
        "direct_wrong_presentations": wrong, "direct_correct_presentations": right,
        "fix_rate_among_direct_wrong": fixes / wrong if wrong else None,
        "harm_rate_among_direct_correct": harms / right if right else None,
        "distinct_sources_with_any_fix": sum(any(not g["direct"]["correct"] and g[a]["correct"] for a in actions) for g in groups),
        "distinct_sources_with_any_harm": sum(any(g["direct"]["correct"] and not g[a]["correct"] for a in actions) for g in groups),
        "diagnostic_anls_changes": {
            "positive": len(positive), "negative": len(negative), "zero": len(gains) - len(positive) - len(negative),
            "mean_positive_change": statistics.fmean(positive) if positive else None,
            "mean_negative_change": statistics.fmean(negative) if negative else None,
            "mean_change": statistics.fmean(gains),
        },
    }


def validate_run(run, manifest, config_path):
    """Fail closed; rebuild all image/chat provenance and independently rescore."""
    run, manifest, config_path = Path(run).resolve(), Path(manifest).resolve(), Path(config_path).resolve()
    metadata = read_json(run / "run.json")
    identity = {key: metadata[key] for key in IDENTITY_KEYS}
    require(canonical_hash(identity) == metadata.get("fingerprint"), "Run fingerprint does not match identity")
    config = metadata["config"]
    require(config == read_json(config_path), "Run config differs from supplied locked config")
    validate_config(config)
    require(metadata.get("experiment") == "native_detail_v3" and metadata.get("development_only") is True,
            "Unexpected experiment identity")
    require(metadata["role"] == "main", "Only the complete locked main panel may produce this report")
    require(metadata.get("base_code_sha256") == BASE_SOURCE_SHA256 == code_digest(), "Archived source digest mismatch")
    require(metadata["code_sha256"] == experiment_digest(), "Inference source files differ from the run")
    pins = config["protocol"]
    if "inference_code_sha256" in pins:
        require(pins["inference_code_sha256"] == metadata["code_sha256"], "Config inference-code pin mismatch")
    if "prompts" in config:
        require(config["prompts"] == {"answer": ANSWER_INSTRUCTION, "crop": CROP_PROMPT, "repeat": REPEAT_PROMPT},
                "Config prompts differ from inference constants")
    require(metadata["model"]["model_id"] == config["model_id"]
            and metadata["model"]["revision"] == pins["model_revision"], "Model identity mismatch")
    require(datetime.fromisoformat(pins["rules_locked_at_utc"].replace("Z", "+00:00"))
            <= datetime.fromisoformat(metadata["created_utc"].replace("Z", "+00:00")),
            "Run predates protocol lock")
    for key, value in metadata["runtime"].items():
        require(metadata.get(key) == value, f"Inconsistent runtime field {key}")
    require(sha256_file(manifest) == metadata["manifest_sha256"] == pins["main_manifest_sha256"],
            "Manifest hash differs from frozen identity")
    sources = validate_manifest(manifest)
    expected_n = pins["main_examples"]
    require(type(expected_n) is int and expected_n > 0 and len(sources) == expected_n, "Manifest cohort size mismatch")
    require(metadata["example_ids"] == [source["example_id"] for source in sources], "Planned IDs/order differ from manifest")
    expected = {(source["example_id"], action) for source in sources for action in ACTIONS}
    records = read_jsonl(run / "records.jsonl")
    require(len(records) == len(expected), f"Expected exactly {len(expected)} complete records; found {len(records)}")
    by_key = {}
    for record in records:
        key = record.get("example_id"), record.get("action")
        require(key in expected and key not in by_key and record.get("status") == "ok",
                f"Unexpected, duplicate or failed record: {key}")
        by_key[key] = record
    require(set(by_key) == expected, "Incomplete action coverage")
    completion = read_json(run / "completed.json")
    require(completion.get("records") == len(expected), "Completion marker count mismatch")
    require(completion.get("records_sha256") == sha256_file(run / "records.jsonl"), "Completion record hash mismatch")
    number(completion.get("elapsed_s"), "study elapsed")
    groups = []
    for source in sources:
        group = {action: by_key[(source["example_id"], action)] for action in ACTIONS}
        with Image.open(manifest.parent / source["image_path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        initial = group["direct"]["response"]
        for action, row in group.items():
            for key, wanted in (("image_id", source["image_id"]), ("source_id", source["source_id"]),
                                ("question", source["question"]), ("target_answer", source["answer"]),
                                ("source_image_sha256", source["image_sha256"])):
                require(row.get(key) == wanted, f"Record/manifest mismatch: {source['example_id']}/{action}/{key}")
            _, images, geometry = build_request(image, source["question"], action, config, initial)
            for key, wanted in geometry.items():
                require(row.get(key) == wanted, f"Reconstructed provenance mismatch: {source['example_id']}/{action}/{key}")
            wanted_grids = [[1, view.height // 16, view.width // 16] for view in images]
            require(row.get("image_grid_thw") == wanted_grids, "Unexpected processor grid")
            visual = sum(t * h * w // 4 for t, h, w in wanted_grids)
            require(type(row.get("visual_tokens")) is int and row["visual_tokens"] == visual, "Visual token count mismatch")
            for key in ("input_tokens", "generated_tokens"):
                require(type(row.get(key)) is int and row[key] > 0, f"Invalid {key}")
            require(row["input_tokens"] > visual, "Input token count must include textual context")
            require(row["generated_tokens"] <= config["answer_max_tokens"], "Exceeded generation limit")
            require(type(row.get("generation_truncated")) is bool, "Invalid truncation flag")
            require(not row["generation_truncated"] or row["generated_tokens"] == config["answer_max_tokens"],
                    "Truncation flag inconsistent with token limit")
            require(row.get("answer_prefix_prefilled") is True and isinstance(row.get("raw_continuation"), str)
                    and row["response"] == "ANSWER:" + row["raw_continuation"], "Response/prefix provenance mismatch")
            for key in ("elapsed_s", "peak_memory_gib", "peak_reserved_gib"):
                number(row.get(key), key)
            require(row["elapsed_s"] > 0 and row["peak_reserved_gib"] >= row["peak_memory_gib"],
                    "Invalid elapsed or allocated/reserved memory ordering")
            expected_scores = score_response(row["response"], source["answer"])
            expected_scores.update(score_official(expected_scores["predicted_answer"] or "", source["answer"]))
            for key, wanted in expected_scores.items():
                require(type(row.get(key)) is type(wanted) and row[key] == wanted,
                        f"Stored score differs from recomputation: {source['example_id']}/{action}/{key}")
            for metric in METRICS:
                number(row[metric], metric)
                require(row[metric] <= 1, "Metric outside [0,1]")
        direct_hash = group["direct"]["image_rgb_sha256"][0]
        for action in ACTIONS:
            if action not in ("direct", "highres"):
                require(group[action]["image_rgb_sha256"][0] == direct_hash, "Follow-up first image differs")
        require(group["repeat"]["image_rgb_sha256"] == [direct_hash, direct_hash], "Repeat is not identical")
        for region in REGIONS:
            first, second = group["native_" + region], group["degraded_" + region]
            for key in ("messages_sha256", "initial_answer_sha256", "overview_size", "additional_size",
                        "source_roi_pixels", "projected_overview_roi_pixels", "image_grid_thw",
                        "input_tokens", "visual_tokens", "followup_prompt"):
                require(first[key] == second[key], f"Native/degraded pair mismatch: {key}")
        groups.append(group)
    return metadata, sources, groups, completion


def gate_decision(primary, oracle, conditions, delta):
    proceed = primary["difference"] >= delta and primary["ci95"][0] > 0 and oracle["difference"] >= delta
    stop = primary["ci95"][1] < delta and oracle["ci95"][1] < delta
    native, direct, highres = (conditions[key] for key in ("native", "direct", "highres"))
    metric = "conservative_text_em"
    flags = {
        "fixed_native_below_direct": native["metrics"][metric] < direct["metrics"][metric],
        "matched_oracle_gap_below_delta": oracle["difference"] < delta,
        "highres_dominates_fixed_native_point_estimates":
            highres["metrics"][metric] >= native["metrics"][metric]
            and highres["full_path_latency_s"]["mean"] <= native["full_path_latency_s"]["mean"],
    }
    # The protocol explicitly allows competing criteria; expose every rule.
    if stop:
        decision = "stop_this_setup"
    elif proceed:
        decision = "revise_before_larger_study" if any(flags.values()) else "proceed_to_larger_untouched_study"
    elif primary["difference"] > 0 and any(flags.values()):
        decision = "revise_or_collect_separately_committed_evidence"
    else:
        decision = "inconclusive"
    return {"decision": decision, "practical_delta": delta, "proceed_headroom_rule_met": proceed,
            "stop_rule_met": stop, "revision_flags": flags,
            "interpretation": "Development decision support, not trained-policy performance or proof of learnability; all rules and competing point-estimate considerations are exposed."}


def summarize(groups, config):
    analysis = config["analysis"]
    require(analysis["primary_metric"] == "conservative_text_em", "Unexpected primary metric")
    samples, seed = analysis["bootstrap_samples"], analysis["bootstrap_seed"]
    require(type(samples) is int and samples > 0 and type(seed) is int, "Invalid bootstrap configuration")
    delta = number(analysis["practical_delta"], "practical delta")
    require(0 < delta < 1, "Practical delta must be between zero and one")
    n = len(groups)
    require(n > 0, "No complete groups")
    rng = random.Random(seed)
    draws = [[rng.randrange(n) for _ in range(n)] for _ in range(samples)]
    conditions = {name: summarize_actions(groups, actions) for name, actions in CONDITIONS.items()}
    action_rows = {action: summarize_actions(groups, [action]) for action in ACTIONS}
    vectors = {metric: {name: [statistics.fmean(group[action][metric] for action in actions) for group in groups]
                        for name, actions in CONDITIONS.items()} for metric in METRICS}
    contrasts = {name: {"role": "primary" if name == "native_minus_degraded" else "secondary",
                       "positive": positive, "negative": negative,
                       "metrics": {metric: paired_contrast(vectors[metric][positive], vectors[metric][negative], draws)
                                   for metric in METRICS}}
                 for name, positive, negative in CONTRASTS}
    candidates = {"native": ["direct", "repeat", *CONDITIONS["native"]],
                  "degraded": ["direct", "repeat", *CONDITIONS["degraded"]],
                  "union": ["direct", "repeat", *CONDITIONS["native"], *CONDITIONS["degraded"]]}
    oracle_vectors = {metric: {name: [max(group[a][metric] for a in actions) for group in groups]
                              for name, actions in candidates.items()} for metric in METRICS}
    oracles = {}
    for name, actions in candidates.items():
        selections = []
        for group in groups:
            best = max(group[action]["conservative_text_em"] for action in actions)
            tied = [action for action in actions if group[action]["conservative_text_em"] == best]
            selections.append(min(tied, key=lambda action: (full_cost(group, action), ACTIONS.index(action))))
        oracles[name] = {
            "candidate_actions": actions,
            "metrics": {metric: statistics.fmean(oracle_vectors[metric][name]) for metric in METRICS},
            "primary_selected_action_counts": {a: selections.count(a) for a in actions},
            "optimistic_primary_selected_full_path_latency_s": describe([full_cost(g, a) for g, a in zip(groups, selections)]),
            "gain_beyond_highres": {metric: statistics.fmean(oracle_vectors[metric][name]) - statistics.fmean(vectors[metric]["highres"]) for metric in METRICS},
            "gain_beyond_best_direct_or_repeat": {
                metric: statistics.fmean(oracle_vectors[metric][name])
                - statistics.fmean(max(g["direct"][metric], g["repeat"][metric]) for g in groups)
                for metric in METRICS},
        }
    oracle_contrasts = {metric: paired_contrast(oracle_vectors[metric]["native"], oracle_vectors[metric]["degraded"], draws)
                        for metric in METRICS}
    primary = contrasts["native_minus_degraded"]["metrics"]["conservative_text_em"]
    return {
        "conditions": conditions, "actions": action_rows, "contrasts": contrasts,
        "changes_vs_direct": {name: changes_vs_direct(groups, actions) for name, actions in CONDITIONS.items()},
        "action_changes_vs_direct": {action: changes_vs_direct(groups, [action]) for action in ACTIONS},
        "privileged_oracles": {
            "label": "Diagnostic privileged label-based upper bounds; not feasible policies.",
            "metric_selection": "Each metric has its own maximum; reported metric maxima need not describe one common selected response.",
            "cost_warning": "Selected costs use reference-guided choice and omit selector/search costs; they are optimistic diagnostics.",
            "oracles": oracles, "matched_native_minus_degraded": oracle_contrasts,
            "native_correct_when_direct_repeat_highres_wrong": sum(
                not any(g[a]["correct"] for a in ("direct", "repeat", "highres"))
                and any(g[a]["correct"] for a in CONDITIONS["native"]) for g in groups),
        },
        "bootstrap": {"samples": samples, "seed": seed, "unit": "original_source_report",
                      "n_source_reports": n, "retains_all_actions_and_quadrants": True,
                      "interval": "95% percentile, exploratory and unadjusted",
                      "resampling": "random.Random(seed).randrange(n), shared paired draws for all metrics/contrasts"},
        "gate": gate_decision(primary, oracle_contrasts["conservative_text_em"], conditions, delta),
    }


def quality_sensitivity(mask_path, metadata, sources, groups):
    """Apply only an outcome-blind mask pinned before main inference; keep primary intact."""
    config = metadata["config"]
    pinned = config["analysis"].get("quality_mask_sha256")
    if mask_path is None:
        require(not pinned, "A quality mask is pinned; supply --quality-mask")
        return None
    require(isinstance(pinned, str) and len(pinned) == 64, "Cannot apply an unpinned quality mask")
    path = Path(mask_path).resolve()
    require(sha256_file(path) == pinned, "Quality mask SHA256 mismatch")
    mask = read_json(path)
    require(mask.get("frozen_before_main_inference") is True, "Mask lacks pre-inference audit declaration")
    require(mask.get("review_is_before_main_inference") is True
            and mask.get("model_predictions_or_outcomes_accessed_by_reviewers") is False,
            "Mask lacks an outcome-blind source-review declaration")
    require(mask.get("main_manifest_sha256") == metadata["manifest_sha256"], "Quality mask refers to another main manifest")
    locked = config["protocol"].get("source_audit_locked_at_utc")
    require(isinstance(locked, str), "Missing source-audit lock time")
    require(datetime.fromisoformat(locked.replace("Z", "+00:00"))
            <= datetime.fromisoformat(metadata["created_utc"].replace("Z", "+00:00")),
            "Quality-mask lock is after main inference began")
    for field in ("locked_at_utc", "created_utc", "frozen_at_utc"):
        if field in mask:
            require(datetime.fromisoformat(mask[field].replace("Z", "+00:00"))
                    <= datetime.fromisoformat(metadata["created_utc"].replace("Z", "+00:00")),
                    f"Quality-mask {field} is after main inference began")
    categories, details = mask["categories_by_example_id"], mask["mask"]
    main_ids = {source["example_id"] for source in sources}
    require(main_ids <= set(categories) and main_ids <= set(details), "Quality mask does not cover the complete main cohort")
    excluded = mask["secondary_excluded_main_ids"]
    require(isinstance(excluded, list) and len(excluded) == len(set(excluded)) and set(excluded) <= main_ids,
            "Invalid quality exclusion IDs")
    eligible_exclusions = {key for key in main_ids if categories[key] in ("label_error", "ambiguous")}
    require(set(excluded) == eligible_exclusions, "Excluded IDs differ from predeclared quality categories")
    source_by_id = {source["example_id"]: source for source in sources}
    for key in main_ids:
        require(details[key]["category"] == categories[key], "Inconsistent quality categories")
        require(details[key].get("cohort") == "main", "Main quality entry has wrong cohort")
        for detail_key, source_key in (("source_id", "source_id"), ("source_image_sha256", "image_sha256"),
                                       ("question", "question"), ("reference_answer", "answer")):
            require(details[key].get(detail_key) == source_by_id[key][source_key],
                    f"Quality-mask source binding mismatch: {key}/{detail_key}")
        if key in eligible_exclusions:
            require(details[key].get("full_page_visual_check") is True,
                    "Excluded source lacks a full-page visual check")
            require(isinstance(details[key].get("reason"), str) and bool(details[key]["reason"].strip()),
                    "Excluded source lacks an audit reason")
    retained = [group for group in groups if group["direct"]["example_id"] not in eligible_exclusions]
    require(bool(retained), "Quality mask excludes every source report")
    result = summarize(retained, config)
    result.pop("gate")
    result.update({"role": "Prespecified secondary source-audit sensitivity; primary full panel is unchanged",
                   "mask_sha256": pinned, "source_audit_locked_at_utc": locked,
                   "primary_n_source_reports": len(groups), "retained_n_source_reports": len(retained),
                   "excluded_n_source_reports": len(excluded), "excluded_example_ids": excluded,
                   "exclusions": {key: details[key] for key in excluded},
                   "gate_applied": False,
                   "caveat": "Eligibility is source-audit-selected, not a random subset. No gate or primary claim is replaced by this sensitivity."})
    return result


def analyze(run, manifest, config_path=None, quality_mask=None):
    run, manifest = Path(run).resolve(), Path(manifest).resolve()
    config_path = Path(config_path or PROJECT / "configs" / "native_detail.json").resolve()
    metadata, sources, groups, completion = validate_run(run, manifest, config_path)
    result = {
        "status": "complete", "experiment": "native_detail_v3", "development_only": True,
        "coverage": {"source_reports": len({r["source_id"] for r in sources}), "images": len(sources),
                     "examples": len(groups), "actions_per_example": len(ACTIONS), "records": len(groups) * len(ACTIONS),
                     "missing_records": 0, "duplicate_records": 0},
        "bindings": {
            "run_fingerprint": metadata["fingerprint"], "run_json_sha256": sha256_file(run / "run.json"),
            "records_sha256": sha256_file(run / "records.jsonl"), "completion_sha256": sha256_file(run / "completed.json"),
            "manifest_sha256": sha256_file(manifest), "locked_config_file_sha256": sha256_file(config_path),
            "config_canonical_sha256": canonical_hash(metadata["config"]),
            "inference_code_sha256": metadata["code_sha256"], "base_source_sha256": metadata["base_code_sha256"],
            "analysis_script_sha256": sha256_file(__file__), "official_metric_upstream_commit": UPSTREAM_COMMIT,
            "official_metric_upstream_sha256": UPSTREAM_SHA256,
        },
        "model": metadata["model"], "runtime": metadata["runtime"], "completion": completion,
        "integrity": {
            "passed": True, "scores_recomputed": True, "all_view_pixels_and_chat_reconstructed": True,
            "description": "Verified source PNG files, rebuilt all request geometry/pixel/chat hashes, checked realized grids and paired input tokens, and recomputed all stored grades. This does not rerun VLM inference.",
            "eligibility_scope": "Full annotation/PDF eligibility is bound through the locked manifest and preparation audit; it is not independently recreated by this image-only analyzer.",
        },
        "cost_definition": metadata["cost_definition"],
        "caveats": [
            "One selected extraction question per original report in a training-release development panel; not held-out transfer.",
            "Native labels a fixed-DPI PDF rendering, not intrinsic photographic pixels; filtering/resampling are part of the intervention.",
            "Four-region means are expected scores of uniformly choosing one region, not acquiring all four.",
            "Native/degraded use identical text and dimensions; native/repeat changes both truthful wording and visual content.",
            "Conservative text EM is the primary custom metric. Official empty-scale single-span EM/F1 are secondary; ANLS is diagnostic.",
            "Invalid parses and truncated responses remain in denominators. Upstream annotation ambiguity and metric normalization can affect scores.",
            "Bootstrap units are original reports; shared company/template effects can remain. Intervals are exploratory and unadjusted.",
            "Direct/highres cost one standalone call. Follow-ups cost direct plus incremental work; reserved memory reflects allocator state.",
            "All timing outliers are retained; synthetic warmup does not exhaust real shapes. PDF rendering, loading and warmup are excluded.",
            "Privileged oracles select using reference labels; neither their quality nor optimistic selected cost is achieved policy performance.",
        ],
    }
    result.update(summarize(groups, metadata["config"]))
    result["source_quality_sensitivity"] = quality_sensitivity(quality_mask, metadata, sources, groups)
    if result["source_quality_sensitivity"]:
        result["bindings"]["source_quality_mask_sha256"] = result["source_quality_sensitivity"]["mask_sha256"]
    return result


def markdown(result):
    c = result["coverage"]
    lines = ["# LookAgain v3: document-detail development pilot", "",
             f"Complete locked panel: **{c['source_reports']} original reports**, **{c['records']} records**, **{c['actions_per_example']} actions per report**.",
             "All grades, source image hashes and reconstructed view/chat provenance passed strict checks. No VLM inference was repeated.", "",
             f"Completion marker reports {result['completion']['elapsed_s'] / 60:.2f} minutes after warmup for the completed execution segment; "
             "this excludes preprocessing/loading and must not be read as total wall time across resumed segments.", "",
             "## Condition means", "",
             "Native/degraded scores average four regions within each report, then reports equally. Those four presentations are paired, not independent reports.", "",
             "| Condition | Conservative EM | Official span EM | Official span F1 | Diagnostic ANLS | Invalid / truncated |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in result["conditions"].items():
        values = " | ".join(f"{100 * row['metrics'][m]:.2f}%" for m in METRICS)
        lines.append(f"| {name} | {values} | {row['invalid_answers']} / {row['truncated_generations']} |")
    lines += ["", "## Paired contrasts", "",
              f"95% paired source-report bootstrap intervals; {result['bootstrap']['samples']:,} samples, seed {result['bootstrap']['seed']}. All intervals are exploratory and unadjusted.",
              "Only native minus degraded on conservative EM is the primary contrast; other rows/metrics are secondary.", "",
              "| Contrast | Metric | Difference, pp | 95% interval, pp |",
              "| --- | --- | ---: | ---: |"]
    for name, row in result["contrasts"].items():
        for metric, stat in row["metrics"].items():
            low, high = stat["ci95_pp"]
            lines.append(f"| {name} | {metric} | {stat['difference_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
    lines += ["", "## Repairs and harms relative to direct", "",
              "| Condition | Presentations | Fixes / direct-wrong | Harms / direct-correct | Reports with any fix / harm | ANLS up / down / same |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in result["changes_vs_direct"].items():
        d = row["diagnostic_anls_changes"]
        lines.append(f"| {name} | {row['n_presentations']} | {row['fixes']} / {row['direct_wrong_presentations']} | "
                     f"{row['harms']} / {row['direct_correct_presentations']} | {row['distinct_sources_with_any_fix']} / {row['distinct_sources_with_any_harm']} | "
                     f"{d['positive']} / {d['negative']} / {d['zero']} |")
    lines += ["", "## All fixed actions and full-path cost", "",
              "Highres/direct are standalone; every follow-up includes direct plus that invocation. No timing outliers are removed.", "",
              "| Action | EM | Official EM / F1 | ANLS | Seconds mean / median / p95 | Allocated GiB mean / max | Reserved GiB mean / max |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in result["actions"].items():
        m, t, a, r = row["metrics"], row["full_path_latency_s"], row["sequential_peak_allocated_gib"], row["sequential_peak_reserved_gib"]
        lines.append(f"| {name} | {100*m['conservative_text_em']:.2f}% | {100*m['official_em']:.2f}% / {100*m['official_f1']:.2f}% | {100*m['anls']:.2f}% | "
                     f"{t['mean']:.3f} / {t['median']:.3f} / {t['p95']:.3f} | {a['mean']:.2f} / {a['max']:.2f} | {r['mean']:.2f} / {r['max']:.2f} |")
    lines += ["", "| Action | Input tokens invocation / full policy, mean | Visual tokens invocation / full policy, mean | Generated tokens invocation / full policy, mean |",
              "| --- | ---: | ---: | ---: |"]
    for name, row in result["actions"].items():
        fields = [f"{row['tokens'][key]['invocation']['mean']:.1f} / {row['tokens'][key]['full_policy']['mean']:.1f}" for key in ("input_tokens", "visual_tokens", "generated_tokens")]
        lines.append("| " + name + " | " + " | ".join(fields) + " |")
    lines += ["", "## Privileged oracle diagnostics", "",
              "These use reference labels to choose an answer and are not deployable controllers. Native/degraded candidate sets each contain direct, repeat and four regions. Highres is excluded from selection.",
              "Every metric has a separate maximum; optimistic selected latency uses the primary EM tie-break and excludes selector/search costs.", "",
              "| Oracle | EM | Official EM / F1 | ANLS | EM gain beyond highres, pp | Optimistic selected mean seconds |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    oracle = result["privileged_oracles"]
    for name, row in oracle["oracles"].items():
        m = row["metrics"]
        lines.append(f"| {name} | {100*m['conservative_text_em']:.2f}% | {100*m['official_em']:.2f}% / {100*m['official_f1']:.2f}% | "
                     f"{100*m['anls']:.2f}% | {100*row['gain_beyond_highres']['conservative_text_em']:+.2f} | "
                     f"{row['optimistic_primary_selected_full_path_latency_s']['mean']:.3f} |")
    lines += ["", "| Matched native minus degraded oracle | Difference, pp | 95% interval, pp |", "| --- | ---: | ---: |"]
    for metric, row in oracle["matched_native_minus_degraded"].items():
        lo, hi = row["ci95_pp"]
        lines.append(f"| {metric} | {row['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    sensitivity = result.get("source_quality_sensitivity")
    if sensitivity:
        lines += ["", "## Secondary source-quality sensitivity", "", sensitivity["role"] + ".",
                  f"Retained {sensitivity['retained_n_source_reports']} of {sensitivity['primary_n_source_reports']} source reports; "
                  f"excluded {sensitivity['excluded_n_source_reports']} from a source-only full-page audit frozen before main inference.",
                  f"Mask SHA256: `{sensitivity['mask_sha256']}`. " + sensitivity["caveat"], "",
                  "| Condition | Conservative EM | Official span EM | Official span F1 | Diagnostic ANLS |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name, row in sensitivity["conditions"].items():
            values = " | ".join(f"{100 * row['metrics'][m]:.2f}%" for m in METRICS)
            lines.append(f"| {name} | {values} |")
        lines += ["", "| Secondary subset contrast | Metric | Difference, pp | 95% interval, pp |",
                  "| --- | --- | ---: | ---: |"]
        for label, stats in (("native minus degraded", sensitivity["contrasts"]["native_minus_degraded"]["metrics"]),
                             ("matched privileged oracle", sensitivity["privileged_oracles"]["matched_native_minus_degraded"])):
            for metric, row in stats.items():
                low, high = row["ci95_pp"]
                lines.append(f"| {label} | {metric} | {row['difference_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
    gate = result["gate"]
    lines += ["", "## Development gate", "", f"Decision support: **{gate['decision']}**; fixed threshold {100*gate['practical_delta']:.2f} pp.",
              f"Proceed headroom rule: {gate['proceed_headroom_rule_met']}. Stop rule: {gate['stop_rule_met']}.",
              "Revision flags: " + "; ".join(f"{k}={v}" for k, v in gate["revision_flags"].items()) + ".",
              gate["interpretation"], "", "## Interpretation limits", ""]
    lines += ["- " + text for text in result["caveats"]]
    lines += ["", "## Reproducibility bindings", ""]
    lines += [f"- {key}: `{value}`." for key, value in result["bindings"].items() if isinstance(value, str)]
    lines += ["", "JSON includes all per-action metrics, token distributions, costs, changes, candidate sets and hash bindings.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "native_detail.json")
    parser.add_argument("--quality-mask", type=Path, help="Optional only when no source-quality hash is pinned in config")
    parser.add_argument("--output", type=Path, help="Defaults to --run; never overwrites raw inputs")
    args = parser.parse_args()
    result = analyze(args.run, args.manifest, args.config, args.quality_mask)
    output = args.output or args.run
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown(result), encoding="utf-8")
    print(f"Wrote complete native-detail analysis to {output}")


if __name__ == "__main__":
    main()
