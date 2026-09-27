"""Independent stdlib-only numerical cross-check of completed v5 results.

Does not import the main analyzer, its helpers, the runner, or model code.
Never reads records before checking that completed.json exists. This verifies
statistics and bindings, not images, grades, semantic truth, or ROI eligibility.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


BUDGETS = (256, 512, 1024)
ACTIONS = tuple(f"{kind}_{budget}" for budget in BUDGETS for kind in ("direct", "native", "degraded")) + ("highres",)
METRICS = ("official_em", "conservative_text_em", "official_f1", "anls")
BINARY = ("official_em", "conservative_text_em")
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260926
IDENTITY = ("config", "manifest_sha256", "code_sha256", "model", "runtime", "role", "example_ids")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite(value, label, lower=0., upper=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= lower
            and (upper is None or value <= upper), "Invalid numeric " + label)
    return value


def percentile(values, fraction):
    """Explicit linear interpolation of ordered observations, like type-7."""
    ordered = sorted(values)
    require(bool(ordered) and 0 <= fraction <= 1, "Invalid percentile input")
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def moments(values):
    require(bool(values), "Empty moments")
    return {"n": len(values), "mean": math.fsum(values) / len(values),
            "median": statistics.median(values), "p95": percentile(values, .95),
            "min": min(values), "max": max(values)}


def draw_reports(count):
    require(type(count) is int and count > 0, "Empty source cohort")
    generator = random.Random(BOOTSTRAP_SEED)
    return [[generator.randrange(count) for _ in range(count)] for _ in range(BOOTSTRAP_SAMPLES)]


def contrast_from_deltas(deltas, draws):
    require(bool(deltas) and all(math.isfinite(x) for x in deltas), "Invalid contrast values")
    count = len(deltas)
    replicates = [math.fsum(deltas[index] for index in indices) / count for indices in draws]
    interval = [percentile(replicates, .025), percentile(replicates, .975)]
    average = math.fsum(deltas) / count
    return {"n_source_reports": count, "difference": average, "difference_pp": 100 * average,
            "ci95": interval, "ci95_pp": [100 * x for x in interval]}


def transition_counts(before, after):
    require(len(before) == len(after) > 0 and all(v in (0., 1.) for v in before + after), "Invalid binary transition input")
    cc = sum(b == 1 and a == 1 for b, a in zip(before, after))
    ww = sum(b == 0 and a == 0 for b, a in zip(before, after))
    fixes = sum(b == 0 and a == 1 for b, a in zip(before, after))
    harms = sum(b == 1 and a == 0 for b, a in zip(before, after))
    correct, wrong, total = cc + harms, ww + fixes, len(before)
    return {"n_source_reports": total, "fixes": fixes, "harms": harms,
            "negative_correct": correct, "negative_wrong": wrong,
            "correct_to_correct": cc, "wrong_to_wrong": ww,
            "fix_rate_among_negative_wrong": fixes / wrong if wrong else None,
            "harm_rate_among_negative_correct": harms / correct if correct else None,
            "fix_fraction_all_reports": fixes / total, "harm_fraction_all_reports": harms / total}


def independent_cost(group, action, field, decision_state=False):
    row_value = group[action][field]
    if not decision_state or action == "highres" or action.startswith("direct_"):
        return row_value
    direct_value = group["direct_" + action.rsplit("_", 1)[1]][field]
    return max(row_value, direct_value) if field in ("peak_memory_gib", "peak_reserved_gib") else row_value + direct_value


def calculate(groups):
    require(bool(groups) and all(set(group) == set(ACTIONS) for group in groups), "Complete ten-action source groups required")
    n = len(groups)
    draws = draw_reports(n)
    vectors = {m: {a: [g[a][m] for g in groups] for a in ACTIONS} for m in METRICS}
    action_stats = {}
    for action in ACTIONS:
        rows = [g[action] for g in groups]
        stats = {"n_source_reports": n, "n_presentations": n,
                 "metrics": {m: math.fsum(vectors[m][action]) / n for m in METRICS},
                 "invalid_answers": sum(not r["parse_valid"] for r in rows),
                 "truncated_generations": sum(r["generation_truncated"] for r in rows),
                 "invocation_latency_s": moments([r["elapsed_s"] for r in rows])}
        for mode in ("standalone", "decision_state"):
            for field, name in (("elapsed_s", "latency_s"), ("peak_memory_gib", "peak_allocated_gib"), ("peak_reserved_gib", "peak_reserved_gib")):
                stats[mode + "_" + name] = moments([independent_cost(g, action, field, mode == "decision_state") for g in groups])
        stats["tokens"] = {field: {mode: moments([independent_cost(g, action, field, mode == "decision_state") for g in groups])
                                  for mode in ("standalone", "decision_state")}
                           for field in ("input_tokens", "visual_tokens", "generated_tokens")}
        action_stats[action] = stats
    effects, transitions, direct_changes = {}, {}, {}
    for budget in BUDGETS:
        native, degraded, direct = f"native_{budget}", f"degraded_{budget}", f"direct_{budget}"
        effects[str(budget)] = {m: contrast_from_deltas([a - b for a, b in zip(vectors[m][native], vectors[m][degraded])], draws) for m in METRICS}
        transitions[str(budget)] = {m: transition_counts(vectors[m][degraded], vectors[m][native]) for m in BINARY}
        for action in (native, degraded):
            direct_changes[action] = {m: transition_counts(vectors[m][direct], vectors[m][action]) for m in BINARY}
    interaction = {m: contrast_from_deltas([
        g["native_256"][m] - g["degraded_256"][m] - g["native_1024"][m] + g["degraded_1024"][m]
        for g in groups], draws) for m in METRICS}
    changed_primary = [i for i, g in enumerate(groups) if g["native_256"]["predicted_answer"] != g["degraded_256"]["predicted_answer"]]
    invalid_raw_changed = [i for i, g in enumerate(groups)
                           if g["native_256"]["response"] != g["degraded_256"]["response"]
                           and not (g["native_256"]["parse_valid"] and g["degraded_256"]["parse_valid"])]
    audit_required = sorted(set(changed_primary) | set(invalid_raw_changed))
    return {"n_source_reports": n, "actions": action_stats, "detail_effects": effects,
            "detail_256_minus_detail_1024": interaction,
            "paired_native_degraded_transitions": transitions,
            "changes_vs_matched_direct": direct_changes,
            "primary_pair_parsed_answer_changed_source_indices": changed_primary,
            "primary_pair_semantic_audit_required_source_indices": audit_required,
            "sum_measured_action_elapsed_s": math.fsum(g[a]["elapsed_s"] for g in groups for a in ACTIONS),
            "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "unit": "original_source_report"}}


def compare_numeric_tree(expected, actual, path, checks):
    """Check every expected leaf; actual may contain descriptive extra fields."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), "Missing object: " + path)
        for key, value in expected.items():
            require(key in actual, "Missing summary field: " + path + "/" + key)
            compare_numeric_tree(value, actual[key], path + "/" + key, checks)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "List mismatch: " + path)
        for index, value in enumerate(expected):
            compare_numeric_tree(value, actual[index], path + f"/{index}", checks)
    else:
        if type(expected) in (int, float):
            require(type(actual) in (int, float) and math.isfinite(actual)
                    and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12),
                    f"Numerical mismatch at {path}: independent={expected!r}, summary={actual!r}")
        else:
            require(type(actual) is type(expected) and actual == expected, f"Value mismatch at {path}")
        checks.append(path)


def check(run, summary_path):
    run, summary_path = Path(run).resolve(), Path(summary_path).resolve()
    # This gate intentionally precedes every read of metadata, records or summary.
    require((run / "completed.json").is_file(), "Run is incomplete: completed.json is required before reading records")
    completion = read_json(run / "completed.json")
    require(type(completion.get("records")) is int and completion["records"] > 0, "Invalid completion marker")
    metadata = read_json(run / "run.json")
    require(metadata.get("role") == "main" and metadata.get("experiment") == "evidence_availability_v5"
            and metadata.get("development_only") is True and metadata.get("label_privileged_localizer") is True,
            "Only a completed primary development run is accepted; no smoke accuracy analysis")
    require(all(key in metadata for key in IDENTITY), "Missing run identity")
    fingerprint = hashlib.sha256(json.dumps({k: metadata[k] for k in IDENTITY}, sort_keys=True).encode()).hexdigest()
    require(fingerprint == metadata.get("fingerprint"), "Run fingerprint mismatch")
    config = metadata["config"]
    require(config["actions"] == list(ACTIONS) and config["analysis"]["primary_metric"] == "official_em", "Unexpected experiment estimand")
    require(config["analysis"]["bootstrap_samples"] == BOOTSTRAP_SAMPLES
            and config["analysis"]["bootstrap_seed"] == BOOTSTRAP_SEED, "Unexpected bootstrap plan")
    ids = metadata["example_ids"]
    require(bool(ids) and all(isinstance(x, str) and x for x in ids) and len(ids) == len(set(ids)), "Invalid planned example IDs")
    require(len(ids) == config["protocol"]["main_examples"] and completion["records"] == len(ids) * len(ACTIONS), "Planned/completed coverage mismatch")
    require(sha256(run / "records.jsonl") == completion.get("records_sha256"), "Completion hash mismatch")
    records = [json.loads(line) for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(records) == completion["records"], "Incomplete or excess records")
    by_key, source_ids = {}, {}
    for row in records:
        key = row.get("example_id"), row.get("action")
        require(key[0] in ids and key[1] in ACTIONS and key not in by_key and row.get("status") == "ok", "Unexpected/duplicate/failed record")
        require(isinstance(row.get("source_id"), str) and row["source_id"], "Missing source report")
        source_ids.setdefault(key[0], row["source_id"])
        require(source_ids[key[0]] == row["source_id"], "Within-example source identity mismatch")
        for metric in METRICS:
            finite(row.get(metric), metric, upper=1)
        require(all(row[m] in (0., 1.) for m in BINARY), "Nonbinary EM")
        for field in ("elapsed_s", "peak_memory_gib", "peak_reserved_gib"):
            finite(row.get(field), field)
        require(row["elapsed_s"] > 0 and row["peak_reserved_gib"] >= row["peak_memory_gib"], "Invalid cost measurement")
        for field in ("input_tokens", "visual_tokens", "generated_tokens"):
            require(type(row.get(field)) is int and row[field] > 0, "Invalid token count")
        require(type(row.get("parse_valid")) is bool and type(row.get("generation_truncated")) is bool, "Invalid output flags")
        require("predicted_answer" in row and (row["predicted_answer"] is None or isinstance(row["predicted_answer"], str)), "Invalid parsed answer")
        by_key[key] = row
    require(len(set(source_ids.values())) == len(ids), "Original source reports are not independent IDs")
    require(set(by_key) == {(example_id, action) for example_id in ids for action in ACTIONS}, "Missing planned action")
    groups = [{action: by_key[(example_id, action)] for action in ACTIONS} for example_id in ids]
    summary = read_json(summary_path)
    require(summary.get("status") == "complete" and summary.get("role") == "main" and summary.get("development_only") is True, "Summary is not the completed main analysis")
    require(summary["per_source_example_ids"] == ids and summary["model"] == metadata["model"], "Summary cohort/model differs")
    require(summary.get("primary_metric") == "official_em" and summary.get("primary_contrast") == "native_256_minus_degraded_256", "Summary primary differs")
    for key, value in (("run_fingerprint", fingerprint), ("records_sha256", completion["records_sha256"]),
                       ("run_json_sha256", sha256(run / "run.json")), ("completion_sha256", sha256(run / "completed.json")),
                       ("manifest_sha256", metadata["manifest_sha256"]), ("inference_code_sha256", metadata["code_sha256"])):
        require(summary["bindings"].get(key) == value, "Summary binding differs: " + key)
    result = calculate(groups)
    checks = []
    for name in ("actions", "detail_effects", "paired_native_degraded_transitions", "changes_vs_matched_direct"):
        compare_numeric_tree(result[name], summary[name], name, checks)
    compare_numeric_tree(result["actions"], summary["conditions"], "conditions", checks)
    compare_numeric_tree(result["detail_256_minus_detail_1024"], summary["contrasts"]["detail_256_minus_detail_1024"]["metrics"], "interaction", checks)
    for budget in BUDGETS:
        compare_numeric_tree(result["detail_effects"][str(budget)], summary["contrasts"][f"native_{budget}_minus_degraded_{budget}"]["metrics"], f"detail_contrast/{budget}", checks)
    compare_numeric_tree(result["primary_pair_parsed_answer_changed_source_indices"],
                         summary["paired_native_degraded_response_agreement"]["256"]["parsed_answer_changed_source_indices"], "primary_parsed_change_indices", checks)
    compare_numeric_tree(result["primary_pair_semantic_audit_required_source_indices"],
                         summary["paired_native_degraded_response_agreement"]["256"]["semantic_audit_required_source_indices"], "primary_semantic_audit_indices", checks)
    expected_coverage = {"source_reports": len(ids), "examples": len(ids), "actions_per_example": len(ACTIONS),
                         "records": len(records), "missing_records": 0, "duplicate_records": 0}
    compare_numeric_tree(expected_coverage, summary["coverage"], "coverage", checks)
    if "sum_measured_action_elapsed_s" in completion:
        compare_numeric_tree(result["sum_measured_action_elapsed_s"], completion["sum_measured_action_elapsed_s"], "measured_elapsed_sum", checks)
    result.update(status="passed", checked_at_utc=datetime.now(timezone.utc).isoformat(), numerical_leaf_checks=len(checks),
        check_paths=checks, tolerance={"relative": 1e-12, "absolute": 1e-12},
        primary_pair_parsed_answer_changed_example_ids=[ids[i] for i in result["primary_pair_parsed_answer_changed_source_indices"]],
        primary_pair_semantic_audit_required_example_ids=[ids[i] for i in result["primary_pair_semantic_audit_required_source_indices"]],
        bindings={"run_fingerprint": fingerprint, "run_json_sha256": sha256(run / "run.json"),
            "records_sha256": completion["records_sha256"], "completion_sha256": sha256(run / "completed.json"),
            "summary_sha256": sha256(summary_path), "check_script_sha256": sha256(__file__)},
        independence="Standalone standard-library implementation; imports no project/analyzer/helper/model modules. Reconstructs paired report draws and percentile interpolation independently.",
        scope="Numerical and identity cross-check only. Does not rerun image reconstruction, model inference, upstream grading, source eligibility, or semantic review. Success is not a positive scientific finding or controller gate.")
    return result


def markdown(result):
    lines = ["# Independent v5 numerical cross-check", "",
        f"Status: **{result['status']}**. {result['n_source_reports']} source reports; {result['numerical_leaf_checks']} numerical/structural leaf comparisons matched.", "",
        result["independence"], "", result["scope"], "",
        "| Overview budget | Metric | Native minus degraded, pp | 95% paired interval, pp |", "|---|---|---:|---:|"]
    for budget, metrics in result["detail_effects"].items():
        for metric, row in metrics.items():
            lo, hi = row["ci95_pp"]
            lines.append(f"| {budget} | {metric} | {row['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "| Secondary interaction: D256 minus D1024 | Difference, pp | 95% paired interval, pp |", "|---|---:|---:|"]
    for metric, row in result["detail_256_minus_detail_1024"].items():
        lo, hi = row["ci95_pp"]
        lines.append(f"| {metric} | {row['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "All ten action means, latency/token/memory summaries for both cost perspectives, output flags, repair/harm denominators, and primary changed-answer indices matched. Bootstrap: 10,000 paired report samples, seed 20260926. Full values are in the adjacent JSON.", "",
              f"Records SHA256: `{result['bindings']['records_sha256']}`.", "",
              f"Summary SHA256: `{result['bindings']['summary_sha256']}`.", "",
              f"Independent script SHA256: `{result['bindings']['check_script_sha256']}`.", ""]
    return "\n".join(lines)


def self_test():
    """Closed-form synthetic checks, never opens a research run."""
    draws = draw_reports(2)
    assert len(draws) == 10000 and all(len(x) == 2 for x in draws)
    constant = contrast_from_deltas([1., 1.], draws)
    assert constant["difference"] == 1. and constant["ci95"] == [1., 1.]
    heterogeneous = contrast_from_deltas([0., 1.], draws)
    assert heterogeneous["difference"] == .5 and heterogeneous["ci95"] == [0., 1.]
    # Correlated budget effects [1,0] cancel within report, before resampling.
    interaction = contrast_from_deltas([a-b for a,b in zip([1.,0.], [1.,0.])], draws)
    assert interaction["ci95"] == [0., 0.]
    transitions = transition_counts([0., 0., 1., 1.], [1., 0., 0., 1.])
    assert transitions["fixes"] == transitions["harms"] == transitions["correct_to_correct"] == transitions["wrong_to_wrong"] == 1
    assert transitions["fix_rate_among_negative_wrong"] == transitions["harm_rate_among_negative_correct"] == .5
    assert transition_counts([0.], [1.])["harm_rate_among_negative_correct"] is None
    group = {"direct_256": {"elapsed_s": 2., "peak_memory_gib": 8.}, "native_256": {"elapsed_s": 1., "peak_memory_gib": 6.},
             "highres": {"elapsed_s": 9., "peak_memory_gib": 12.}}
    assert independent_cost(group, "native_256", "elapsed_s") == 1.
    assert independent_cost(group, "native_256", "elapsed_s", True) == 3.
    assert independent_cost(group, "native_256", "peak_memory_gib", True) == 8.
    assert independent_cost(group, "highres", "elapsed_s", True) == 9.
    assert moments([1., 2., 3.])["mean"] == 2. and percentile([0., 10.], .25) == 2.5
    checks = []
    compare_numeric_tree({"x": [1., None]}, {"x": [1. + 1e-14, None]}, "fixture", checks)
    assert len(checks) == 2
    try:
        compare_numeric_tree({"mean": .5}, {"mean": .6}, "tamper", [])
    except ValueError:
        pass
    else:
        raise AssertionError("Numerical tampering was not rejected")
    return {"self_test": "passed", "scope": "Closed-form synthetic math/cost/comparison checks; no research records read"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    if args.run is None or args.summary is None or args.output is None:
        parser.error("--run, --summary and --output are required unless --self-test is used")
    result = check(args.run, args.summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "independent_result_check.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "independent_result_check.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "reports": result["n_source_reports"], "checks": result["numerical_leaf_checks"]}))


if __name__ == "__main__":
    main()
