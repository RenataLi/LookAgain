"""Completed-only DUDE replication statistics; never generates model responses.

The run validator reconstructs frozen source, lock, pixel/chat/processor and
grade provenance before this module calculates anything. Numerical helpers are
also usable on synthetic fixtures; only analyze() assigns a real-run status.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
ACTIONS = ("direct_256", "native_256", "degraded_256", "highres")
METRICS = ("primary_em", "official_anls")
PRIMARY = "native_256_minus_degraded_256"
BOOTSTRAP_SAMPLES = 10000
SEED = 20260927
MAIN_N = 660
CONTRASTS = (
    ("native_256", "degraded_256"),
    ("native_256", "direct_256"),
    ("degraded_256", "direct_256"),
    ("highres", "direct_256"),
    ("native_256", "highres"),
    ("degraded_256", "highres"),
)
MEASUREMENTS = ("elapsed_s", "processor_observer_elapsed_s", "peak_memory_gib", "peak_reserved_gib",
                "input_tokens", "visual_tokens", "generated_tokens")
LABELS = {"direct_256": "Overview 256", "native_256": "Native ROI",
          "degraded_256": "Degraded ROI", "highres": "Full page 4096"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    require(len(values) > 0, "Empty measurement distribution")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values),
            "Nonfinite or nonnumeric measurement")
    return {"n": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "min": min(values),
            "p95": float(np.quantile(values, .95, method="linear")), "max": max(values)}


def exact_mcnemar(native_only, degraded_only):
    """Two-sided exact conditional Binomial(b+c, .5); not mid-p/asymptotic.

    Integer binomial coefficients avoid cancellation in extreme or balanced
    tables. With no discordance, the exact p-value is one.
    """
    require(all(type(x) is int and x >= 0 for x in (native_only, degraded_only)),
            "Discordant counts must be nonnegative integers")
    n = native_only + degraded_only
    k = min(native_only, degraded_only)
    tail_numerator = sum(math.comb(n, j) for j in range(k + 1))
    p = min(1., (2 * tail_numerator) / (1 << n))
    return {"native_only_correct": native_only, "degraded_only_correct": degraded_only,
            "discordant_pairs": n, "p_two_sided": p, "alpha": .05,
            "positive_direction": native_only > degraded_only,
            "positive_direction_rejection": native_only > degraded_only and p <= .05,
            "method": "Two-sided exact conditional binomial McNemar; no mid-p correction",
            "null": "Equal probabilities of the two discordant outcomes"}


def bootstrap_draws(n, samples=BOOTSTRAP_SAMPLES, seed=SEED):
    require(type(n) is int and n > 0 and type(samples) is int and samples > 0
            and type(seed) is int, "Invalid bootstrap parameters")
    rng = random.Random(seed)
    return np.asarray([[rng.randrange(n) for _ in range(n)] for _ in range(samples)],
                      dtype=np.int32)


def paired_interval(positive, negative, draws):
    require(len(positive) == len(negative) and len(positive) > 0,
            "Empty or unpaired contrast")
    delta = np.asarray(positive, dtype=float) - np.asarray(negative, dtype=float)
    require(bool(np.isfinite(delta).all()), "Nonfinite paired score")
    draws = np.asarray(draws)
    require(draws.ndim == 2 and draws.shape[0] > 0 and draws.shape[1] == len(delta)
            and np.issubdtype(draws.dtype, np.integer)
            and bool((draws >= 0).all()) and bool((draws < len(delta)).all()),
            "Invalid paired bootstrap indices")
    means = delta[draws].mean(axis=1)
    interval = np.quantile(means, [.025, .975], method="linear").tolist()
    point = statistics.fmean(delta.tolist())
    return {"n_source_clusters": len(delta), "difference": point,
            "difference_pp": 100 * point, "ci95": interval,
            "ci95_pp": [100 * v for v in interval],
            "degenerate_empirical_interval": interval[0] == interval[1]}


def validate_analysis_plan(config):
    """Do not silently apply different inferential settings from the run."""
    require(tuple(config.get("actions", ())) == ACTIONS, "Four locked actions differ")
    plan = config.get("analysis_plan", {})
    expected = {
        "primary_metric": "source_validated_casefold_whitespace_exact_match",
        "secondary_metric": "pinned_official_DUDE_scalar_ANLS",
        "primary_contrast": PRIMARY,
        "primary_test": "two-sided exact McNemar, positive observed direction",
        "alpha": .05, "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": SEED, "bootstrap_unit": "source_cluster",
        "bootstrap_interval": "paired percentile 95%",
        "planning_delta": .05, "planning_total_discordance": .20,
        "target_power_scenario": .80, "automatic_controller_training": False,
        "quantitative_advancement": [
            "mean(native_256-degraded_256)>=0.02",
            "exact_two_sided_p<=0.05",
            "paired_ci95_lower>0",
            "mean(native_256)>=mean(direct_256)",
        ],
    }
    require(all(plan.get(k) == v and type(plan.get(k)) is type(v)
                for k, v in expected.items()), "Locked statistical plan differs")
    require(config.get("source_plan", {}).get("main_target_source_clusters") == MAIN_N,
            "Locked main target differs")


def validate_table(rows, records):
    """Additional statistic-layer pairing/finite-value guard, not provenance QA."""
    require(bool(rows) and isinstance(records, dict), "Empty source/record table")
    ids = [r["example_id"] for r in rows]
    clusters = [r["source_cluster_id"] for r in rows]
    require(all(isinstance(x, str) and x for x in ids + clusters),
            "Missing source identity")
    require(len(set(ids)) == len(ids) and len(set(clusters)) == len(clusters),
            "Duplicate example or source cluster")
    require(set(records) == {(eid, action) for eid in ids for action in ACTIONS},
            "Complete four-action cohort required; no missing or extra records")
    groups = []
    for source in rows:
        group = {}
        for action in ACTIONS:
            r = records[source["example_id"], action]
            require(r.get("example_id") == source["example_id"]
                    and r.get("source_cluster_id") == source["source_cluster_id"]
                    and r.get("action") == action, "Record identity disagrees with its key")
            require(r.get("status") == "ok", "Failed action cannot enter completed analysis")
            for metric in METRICS:
                value = r.get(metric)
                require(type(value) in (int, float) and math.isfinite(value)
                        and 0 <= value <= 1, "Invalid bounded metric")
            require(r["primary_em"] in (0, 1), "Primary EM must be binary")
            require(type(r.get("parse_valid")) is bool
                    and type(r.get("generation_truncated")) is bool, "Missing format flags")
            require(isinstance(r.get("response"), str), "Missing full response")
            if r["parse_valid"]:
                require(isinstance(r.get("predicted_answer"), str)
                        and bool(r["predicted_answer"].strip())
                        and isinstance(r.get("normalized_prediction"), str)
                        and bool(r["normalized_prediction"].strip()), "Invalid parsed prediction")
            else:
                require(r.get("predicted_answer") is None and r.get("normalized_prediction") is None
                        and all(r[m] == 0 for m in METRICS), "Invalid parse must score zero")
            for field in MEASUREMENTS:
                v = r.get(field)
                require(type(v) in (int, float) and math.isfinite(v) and v >= 0,
                        "Invalid cost/token measurement")
            require(r["elapsed_s"] > 0
                    and r["peak_reserved_gib"] >= r["peak_memory_gib"], "Invalid latency/memory")
            require(r["processor_observer_elapsed_s"] <= r["elapsed_s"],
                    "Observer time exceeds measured invocation time")
            require(all(type(r[f]) is int and r[f] > 0
                        for f in ("input_tokens", "visual_tokens", "generated_tokens"))
                    and r["input_tokens"] > r["visual_tokens"]
                    and r["generated_tokens"] <= 64, "Invalid token counts")
            require(not r["generation_truncated"] or r["generated_tokens"] == 64,
                    "Truncation flag inconsistent with token cap")
            group[action] = r
        groups.append(group)
    return groups


def cost(group, action, field="elapsed_s", mode="standalone"):
    require(mode in ("standalone", "decision_state") and action in ACTIONS
            and field in MEASUREMENTS, "Unknown cost perspective/action/field")
    value = group[action][field]
    if mode == "standalone" or action in ("direct_256", "highres"):
        return value
    initial = group["direct_256"][field]
    return max(value, initial) if field in ("peak_memory_gib", "peak_reserved_gib") else value + initial


def summarize_action(groups, action, *, include_quality):
    records = [g[action] for g in groups]
    result = {"n_source_clusters": len(groups), "n_invocations": len(records),
              "successful_records": len(records), "failed_records": 0,
              "invalid_answers": sum(not r["parse_valid"] for r in records),
              "truncated_generations": sum(r["generation_truncated"] for r in records),
              "valid_truncated_answers": sum(r["generation_truncated"] and r["parse_valid"]
                                             for r in records)}
    result["costs"] = {
        mode: {field: describe([cost(g, action, field, mode) for g in groups])
               for field in MEASUREMENTS}
        for mode in ("standalone", "decision_state")
    }
    result["observations"] = {field: [r[field] for r in records] for field in MEASUREMENTS}
    if include_quality:
        result["metrics"] = {m: statistics.fmean(r[m] for r in records) for m in METRICS}
        result["primary_correct"] = sum(r["primary_em"] == 1 for r in records)
    return result


def transitions(groups, positive, negative):
    before = [g[negative]["primary_em"] for g in groups]
    after = [g[positive]["primary_em"] for g in groups]
    correct = sum(b == 1 for b in before)
    wrong = len(groups) - correct
    fixes = sum(b == 0 and a == 1 for b, a in zip(before, after))
    harms = sum(b == 1 and a == 0 for b, a in zip(before, after))
    return {"metric": "primary_em", "positive": positive, "negative": negative,
            "n_source_clusters": len(groups), "fixes": fixes, "harms": harms,
            "negative_correct": correct, "negative_wrong": wrong,
            "correct_to_correct": correct - harms, "wrong_to_wrong": wrong - fixes,
            "fix_rate_among_negative_wrong": fixes / wrong if wrong else None,
            "harm_rate_among_negative_correct": harms / correct if correct else None,
            "fix_fraction_all_sources": fixes / len(groups),
            "harm_fraction_all_sources": harms / len(groups),
            "interpretation": "Automatic score transitions; not semantic content-repair judgments"}


def pair_diagnostics(rows, groups, positive="native_256", negative="degraded_256"):
    pairs = [(g[positive], g[negative]) for g in groups]
    changed = [i for i, (a, b) in enumerate(pairs) if a["predicted_answer"] != b["predicted_answer"]]
    invalid_raw = [i for i, (a, b) in enumerate(pairs)
                   if a["response"] != b["response"] and not (a["parse_valid"] and b["parse_valid"])]
    audit = sorted(set(changed) | set(invalid_raw))
    by_category = {}
    for label, condition in (
        ("both_correct", lambda a, b: a["primary_em"] == b["primary_em"] == 1),
        ("both_wrong", lambda a, b: a["primary_em"] == b["primary_em"] == 0),
        ("native_only_correct", lambda a, b: a["primary_em"] == 1 and b["primary_em"] == 0),
        ("degraded_only_correct", lambda a, b: a["primary_em"] == 0 and b["primary_em"] == 1),
    ):
        by_category[label] = [rows[i]["example_id"] for i in audit if condition(*pairs[i])]
    return {"n_source_clusters": len(rows),
            "raw_response_equal": sum(a["response"] == b["response"] for a, b in pairs),
            "parsed_answer_equal": len(rows) - len(changed),
            "normalized_prediction_equal": sum(a["normalized_prediction"] == b["normalized_prediction"]
                                                for a, b in pairs),
            "both_parse_valid": sum(a["parse_valid"] and b["parse_valid"] for a, b in pairs),
            "metric_equal": {m: sum(a[m] == b[m] for a, b in pairs) for m in METRICS},
            "parsed_answer_changed_ids": [rows[i]["example_id"] for i in changed],
            "raw_changed_with_invalid_parse_ids": [rows[i]["example_id"] for i in invalid_raw],
            "semantic_audit_required_ids": [rows[i]["example_id"] for i in audit],
            "semantic_audit_required_source_cluster_ids": [rows[i]["source_cluster_id"] for i in audit],
            "audit_cases_by_automatic_primary_score": by_category,
            "agreement_counts_include_invalid_parses": True,
            "audit_scope": "All parsed-answer changes, including equal-grade pairs, plus raw differences when either parse is invalid. None equality can include two invalid parses.",
            "semantic_judgments_inferred": False}


def advancement_checks(primary, test, native_mean, direct_mean):
    checks = {"mean_difference_at_least_0_02": primary["difference"] >= .02,
              "exact_two_sided_p_at_most_0_05": test["p_two_sided"] <= .05,
              "paired_ci95_lower_positive": primary["ci95"][0] > 0,
              "native_mean_at_least_direct_mean": native_mean >= direct_mean}
    return {"checks": checks, "positive_observed_direction": primary["difference"] > 0,
            "quantitative_rule_met": all(checks.values()) and primary["difference"] > 0,
            "semantic_screen_status": "pending_separate_condition_blinded_artifact",
            "semantic_screen_requirements": {"distinct_unambiguous_native_only_repairs_at_least": 3,
                                             "net_content_repairs_positive": True},
            "overall_advancement_status": "not_established_pending_semantic_screen"
                if all(checks.values()) and primary["difference"] > 0 else "quantitative_rule_not_met",
            "automatic_controller_training": False, "controller_ready": False,
            "interpretation": "A passed quantitative rule is not full advancement or learned-policy evidence; the separate descriptive screen cannot rescue a failed quantitative rule."}


def summarize(rows, records, *, engineering=False, samples=BOOTSTRAP_SAMPLES, seed=SEED):
    """Pure synthetic-testable summary; only analyze() validates a real run."""
    groups = validate_table(rows, records)
    n = len(rows)
    actions = {a: summarize_action(groups, a, include_quality=not engineering) for a in ACTIONS}
    result = {"n_source_clusters": n, "n_records": len(records), "actions": actions,
              "example_ids": [r["example_id"] for r in rows],
              "source_cluster_ids": [r["source_cluster_id"] for r in rows],
              "format_diagnostics_by_action": {
                  a: {k: actions[a][k] for k in ("n_invocations", "successful_records",
                     "failed_records", "invalid_answers", "truncated_generations", "valid_truncated_answers")}
                  for a in ACTIONS},
              "standalone_measured_action_time_sum_s": math.fsum(r["elapsed_s"] for r in records.values())}
    if engineering:
        result.update(status="completed_engineering_technical_check",
                      main_scientific_inference_performed=False, quality_scores_reported=False,
                      interpretation="Separate engineering requests only; no main test, confidence interval, gate or quality comparison is reported.")
        return result
    draws = bootstrap_draws(n, samples, seed)
    vectors = {m: {a: [g[a][m] for g in groups] for a in ACTIONS} for m in METRICS}
    for action in ACTIONS:
        actions[action]["metric_ci95"] = {
            m: paired_interval(vectors[m][action], [0.] * n, draws)["ci95"] for m in METRICS
        }
    contrasts = {}
    for pos, neg in CONTRASTS:
        name = pos + "_minus_" + neg
        contrasts[name] = {"positive": pos, "negative": neg,
            "primary_only_for_metric": "primary_em" if name == PRIMARY else None,
            "metrics": {m: paired_interval(vectors[m][pos], vectors[m][neg], draws) for m in METRICS},
            "primary_em_transitions": transitions(groups, pos, neg)}
    primary = contrasts[PRIMARY]["metrics"]["primary_em"]
    transition = contrasts[PRIMARY]["primary_em_transitions"]
    test = exact_mcnemar(transition["fixes"], transition["harms"])
    gate = advancement_checks(primary, test, actions["native_256"]["metrics"]["primary_em"],
                              actions["direct_256"]["metrics"]["primary_em"])
    result.update(primary_metric="primary_em", primary_contrast=PRIMARY,
                  primary_result={**primary, "exact_mcnemar": test},
                  contrasts=contrasts, quantitative_advancement=gate,
                  per_source_action_scores=vectors,
                  native_degraded_response_agreement=pair_diagnostics(rows, groups),
                  bootstrap={"samples": samples, "seed": seed, "unit": "source_cluster",
                             "n_source_clusters": n, "all_four_actions_paired": True,
                             "interval": "95% percentile, linear quantile interpolation",
                             "draw_algorithm": "Shared random.Random(seed).randrange(n), sample-major order",
                             "secondary_intervals_unadjusted": True})
    return result


def validate_completed_run(manifest, lock_path, run_dir, role="main"):
    # Lazy import makes numerical unit tests independent of GPU/runtime packages.
    from dude_replication import validate_completed_run as validate
    return validate(manifest, lock_path, run_dir, role=role)


def analyze(manifest, lock_path, run_dir, *, engineering=False):
    """No records/statistics are read before the strict completion validator."""
    manifest, lock_path, run_dir = map(lambda p: Path(p).resolve(),
                                      (manifest, lock_path, run_dir))
    role = "engineering" if engineering else "main"
    rows, records, run, completed = validate_completed_run(manifest, lock_path, run_dir, role=role)
    require(run.get("role") == role, "Run role differs from explicitly requested role")
    require(engineering or len(rows) == MAIN_N, "Main analysis requires exactly 660 sources")
    validate_analysis_plan(run["config"])
    result = summarize(rows, records, engineering=engineering)
    errors_path = run_dir / "errors.jsonl"
    errors = [json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines()
              if line.strip()] if errors_path.exists() else []
    result.update(
        schema_version=1, experiment="dude_replication",
        status="completed_engineering_technical_check" if engineering else "completed_primary_replication",
        role=role, created_utc=datetime.now(timezone.utc).isoformat(),
        fingerprint=run["fingerprint"], model=run.get("model"), runtime=run.get("runtime"),
        completion=completed, runtime_error_log_entries=len(errors),
        runtime_error_log=errors, errors_not_scored_or_silently_dropped=True,
        privilege={"answer_page_privileged": True, "roi_privileged": True,
                   "learned_selector_evaluated": False, "full_DUDE_benchmark_evaluation": False},
        provenance={
            "manifest_sha256": sha256(manifest), "execution_lock_sha256": sha256(lock_path),
            "run_json_sha256": sha256(run_dir / "run.json"),
            "records_sha256": sha256(run_dir / "records.jsonl"),
            "completed_sha256": sha256(run_dir / "completed.json"),
            "analysis_script_sha256": sha256(Path(__file__)),
            "completion_validator_sha256": sha256(PROJECT / "experiments/dude_replication.py"),
            "metric_adapter_sha256": sha256(PROJECT / "experiments/dude_metrics.py"),
            "errors_sha256": sha256(errors_path) if errors_path.exists() else None,
            "numpy_version": np.__version__,
        },
        integrity={"completion_and_execution_lock_validated_before_statistics": True,
                   "source_pixel_chat_roi_processor_and_score_reconstruction": True,
                   "exact_four_action_source_pairing_validated": True,
                   "statistic_layer_finite_scores_measurements_and_unique_sources_validated": True,
                   "full_validator": "dude_replication.validate_completed_run",
                   "new_model_calls": 0},
        cost_definitions={
            "primary": "Instrumented standalone synchronized invocation on an active desktop; fresh regional calls require no previous answer. This is not isolated throughput or energy measurement.",
            "observer": "CPU processor-input digest validation is included in standalone elapsed time. Its separately reported overhead is not subtracted.",
            "secondary": "Hypothetical decision-state direct_256 plus regional latency/tokens; direct/highres remain standalone.",
            "memory": "Sequential peak is max(direct, regional), never their sum.",
            "excluded": "Preparation, model loading, warmup and logging are separate. Discovering the privileged page/ROI has no measured deployment cost.",
            "wall_time": "Completion metadata is preserved verbatim; summed measured invocations are separate from wall-clock execution segments and resume history.",
        },
        limitations=[
            "Source/page/ROI selection is annotation-privileged; no nonprivileged selector or controller is evaluated.",
            "Source clusters reduce known duplicates but do not prove independence; residual template dependence and pretraining exposure remain unknown.",
            "Primary exact match retains formatting/signs/units; semantic content review cannot change automatic grades.",
            "The exact McNemar test and percentile CI are not inversions; report both even if a boundary decision differs.",
            "A degenerate empirical bootstrap interval does not establish equivalence.",
            "All other action/ANLS contrasts and intervals are descriptive and unadjusted.",
            "Observed gain at least2pp with lower CI above zero does not show the true effect exceeds2pp.",
            "Token ceilings do not establish matched compute, speed or energy savings.",
        ])
    return result


def markdown(result):
    engineering = result["role"] == "engineering"
    lines = ["# DUDE engineering technical check" if engineering else "# DUDE prospective replication",
             "", f"Status: **{result['status']}**. Sources: **{result['n_source_clusters']}**; completed invocations: **{result['n_records']}**.",
             "", "The answer page and evidence ROI are annotation-privileged. No learned selector or full-document DUDE evaluation is claimed.", ""]
    if engineering:
        lines += ["Engineering-only output: primary tests, quality comparisons and advancement decisions are withheld.", ""]
    else:
        p = result["primary_result"]; test = p["exact_mcnemar"]; gate = result["quantitative_advancement"]
        lines += [f"Sole primary comparison: native256 minus degraded256, custom primary EM. Difference **{p['difference_pp']:.3f} pp**, 95% paired percentile CI **[{p['ci95_pp'][0]:.3f}, {p['ci95_pp'][1]:.3f}] pp**.",
                  f"Exact two-sided McNemar p = **{test['p_two_sided']:.8g}**; native-only correct **{test['native_only_correct']}**, degraded-only correct **{test['degraded_only_correct']}**.",
                  f"Quantitative rule met: **{gate['quantitative_rule_met']}**. Overall status: **{gate['overall_advancement_status']}**. The separate condition-blinded content review remains pending.", "",
                  "| Prespecified quantitative criterion | Met |", "|---|---|"]
        lines += [f"| {name} | {value} |" for name, value in gate["checks"].items()]
        lines += ["", "## Quality", "", "| Action | Correct / N | Primary EM, % | Official scalar ANLS |",
                  "|---|---:|---:|---:|"]
        for action in ACTIONS:
            a = result["actions"][action]
            lines.append(f"| {action} | {a['primary_correct']} / {a['n_source_clusters']} | {100*a['metrics']['primary_em']:.3f} | {a['metrics']['official_anls']:.5f} |")
        lines += ["", "## Paired comparisons", "", "Only native256-minus-degraded256 on primary EM is primary. All other intervals are descriptive and unadjusted.", "",
                  "| Contrast | Metric | Difference, pp | 95% paired CI, pp |", "|---|---|---:|---:|"]
        for name, contrast in result["contrasts"].items():
            for metric, s in contrast["metrics"].items():
                lines.append(f"| {name} | {metric} | {s['difference_pp']:.3f} | [{s['ci95_pp'][0]:.3f}, {s['ci95_pp'][1]:.3f}] |")
        lines += ["", "## Automatic transitions", "", "Fix/harm labels below describe exact-match transitions, not semantic content repairs.", "",
                  "| Comparison | Fixes / negative wrong | Harms / negative correct | Correct → correct | Wrong → wrong |",
                  "|---|---:|---:|---:|---:|"]
        for name, contrast in result["contrasts"].items():
            t = contrast["primary_em_transitions"]
            def rate(value):
                return "undefined" if value is None else f"{100*value:.2f}%"
            lines.append(f"| {name} | {t['fixes']} / {t['negative_wrong']} ({rate(t['fix_rate_among_negative_wrong'])}) | {t['harms']} / {t['negative_correct']} ({rate(t['harm_rate_among_negative_correct'])}) | {t['correct_to_correct']} | {t['wrong_to_wrong']} |")
        agreement = result["native_degraded_response_agreement"]
        lines += ["", "## Condition-blinded review still required", "",
                  f"Required cases: **{len(agreement['semantic_audit_required_ids'])}**. The JSON enumerates every parsed-answer change and every raw difference with at least one invalid parse, including equal-grade pairs. No content labels were inferred.",
                  f"Raw response agreements: {agreement['raw_response_equal']}; parsed agreements: {agreement['parsed_answer_equal']}; both parses valid: {agreement['both_parse_valid']}. Equality counts include invalid parses and are not accuracy measures.", ""]
    lines += ["## Format and completion", "", "| Action | Records | Invalid answers | Truncated | Valid truncated |", "|---|---:|---:|---:|---:|"]
    for action, d in result["format_diagnostics_by_action"].items():
        lines.append(f"| {action} | {d['n_invocations']} | {d['invalid_answers']} | {d['truncated_generations']} | {d['valid_truncated_answers']} |")
    lines += ["", f"All analyzed records have successful invocation status. Runtime error-log entries preserved separately: **{result['runtime_error_log_entries']}**. Invalid parses score zero; truncation remains in the cohort.", "",
              "## Measured cost", "", *[f"- **{k}:** {v}" for k, v in result["cost_definitions"].items()], "",
              "| Action | Standalone latency mean / median / p95 / max, s | Decision-state latency mean / median / p95 / max, s | Standalone allocated / reserved peak maxima, GiB |",
              "|---|---:|---:|---:|"]
    for action, a in result["actions"].items():
        s, d = a["costs"]["standalone"], a["costs"]["decision_state"]
        fmt = lambda x: " / ".join(f"{x[k]:.3f}" for k in ("mean", "median", "p95", "max"))
        lines.append(f"| {action} | {fmt(s['elapsed_s'])} | {fmt(d['elapsed_s'])} | {s['peak_memory_gib']['max']:.3f} / {s['peak_reserved_gib']['max']:.3f} |")
    lines += ["", "| Action | Included processor observer overhead mean / median / p95 / max, s |", "|---|---:|"]
    for action, a in result["actions"].items():
        observer = a["costs"]["standalone"]["processor_observer_elapsed_s"]
        lines.append(f"| {action} | " + " / ".join(f"{observer[k]:.6f}" for k in ("mean", "median", "p95", "max")) + " |")
    lines += ["", "| Action | Input tokens mean / max | Visual tokens mean / max | Generated tokens mean / max |", "|---|---:|---:|---:|"]
    for action, a in result["actions"].items():
        s = a["costs"]["standalone"]
        vals = [f"{s[k]['mean']:.2f} / {s[k]['max']}" for k in ("input_tokens", "visual_tokens", "generated_tokens")]
        lines.append(f"| {action} | " + " | ".join(vals) + " |")
    lines += ["", f"Sum of all measured standalone invocations: **{result['standalone_measured_action_time_sum_s']:.3f} s**.",
              "The JSON retains every cost observation, complete completion/runtime metadata, all source-paired scores, and artifact hashes. A completion wall-time field may cover only its documented resume segment.", "",
              "## Integrity and interpretation", "",
              "Completed-run validation reconstructed source/lock, pixels, chat, ROI, processor grids and scores before these statistics. No inference was repeated.",
              "", *[f"- {text}" for text in result["limitations"]], ""]
    if not engineering:
        lines += ["## Figures", "", "![Action metrics](figures/quality.png)", "",
                  "![Paired contrasts](figures/paired_contrasts.png)", "",
                  "![Standalone latency distributions](figures/latency.png)", ""]
    return "\n".join(lines)


def make_figures(result, output):
    require(result["role"] == "main", "Scientific figures require a completed main run")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(output) / "figures"
    out.mkdir()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    colors = ["#596777", "#126782", "#bd6426", "#7851a9"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, metric, title in zip(axes, METRICS, ("Custom primary EM", "Official scalar ANLS (secondary)")):
        for i, action in enumerate(ACTIONS):
            a = result["actions"][action]; mean = a["metrics"][metric]; lo, hi = a["metric_ci95"][metric]
            ax.vlines(i, lo, hi, colors=colors[i], lw=2)
            ax.scatter(i, mean, color=colors[i], s=42, zorder=3)
        ax.set(xticks=range(4), xticklabels=[LABELS[a] for a in ACTIONS], ylim=(-.02, 1.02),
               title=title, ylabel="Mean score")
        ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"DUDE audited subset: {result['n_source_clusters']} paired source clusters\nAnnotation-privileged answer page and ROI; 95% paired bootstrap intervals")
    for ext in ("png", "pdf"): fig.savefig(out / f"quality.{ext}")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    names = list(result["contrasts"])
    for i, name in enumerate(names):
        s = result["contrasts"][name]["metrics"]["primary_em"]
        color = "#126782" if name == PRIMARY else "#737b83"
        ax.hlines(i, *s["ci95_pp"], colors=color, lw=2)
        ax.scatter(s["difference_pp"], i, c=color, s=40, zorder=3)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set(yticks=range(len(names)), yticklabels=[x.replace("_256", "").replace("_minus_", " − ") for x in names],
           xlabel="Custom primary EM difference, percentage points",
           title="Paired contrasts: only native − degraded is primary\nOther intervals are descriptive and unadjusted")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
    for ext in ("png", "pdf"): fig.savefig(out / f"paired_contrasts.{ext}")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for action, color in zip(ACTIONS, colors):
        values = np.sort(result["actions"][action]["observations"]["elapsed_s"])
        ax.step(values, np.arange(1, len(values)+1)/len(values), where="post",
                color=color, label=LABELS[action])
    ax.set(xlabel="Standalone synchronized invocation latency, seconds",
           ylabel="Cumulative fraction of source invocations", ylim=(0, 1.02),
           title="Observed latency distributions\nPrivileged page/ROI discovery, loading and warmup excluded")
    ax.legend(); ax.grid(alpha=.2)
    for ext in ("png", "pdf"): fig.savefig(out / f"latency.{ext}")
    plt.close(fig)
    return {"matplotlib_version": matplotlib.__version__,
            "files": {p.relative_to(Path(output)).as_posix(): sha256(p) for p in sorted(out.iterdir())}}


def write_artifacts(result, output):
    output = Path(output)
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "Analysis output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    if result["role"] == "main":
        result["figures"] = make_figures(result, output)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                    allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown(result), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--engineering", action="store_true",
                   help="Validate engineering role and emit technical-only summary; never a primary conclusion")
    args = p.parse_args(argv)
    require(not args.output.exists() or (args.output.is_dir() and not any(args.output.iterdir())),
            "Analysis output must be absent or empty")
    result = analyze(args.manifest, args.lock, args.run, engineering=args.engineering)
    write_artifacts(result, args.output)
    print(json.dumps({"status": result["status"], "sources": result["n_source_clusters"],
                      "records": result["n_records"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
