"""Prospective exact paired-binary power. Standard library; no model-data reads.

Example: python experiments/plan_replication_power.py --output reports/replication_plan
The search ceiling is a numerical planning range, never an inference budget.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

ALPHA = 0.05
EFFECTS = (0.03, 0.035, 0.05)
DISCORDANCES = (0.08, 0.15, 0.20)


def _validate_n(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")


def binomial_pmf(n, p):
    """Normalized mode-centered recurrence, avoiding factorials and tiny starts."""
    _validate_n(n)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be finite and in [0, 1]")
    if p == 0:
        return [1.0] + [0.0] * n
    if p == 1:
        return [0.0] * n + [1.0]
    mode = min(n, math.floor((n + 1) * p))
    weights = [0.0] * (n + 1)
    weights[mode] = 1.0
    for k in range(mode, 0, -1):
        weights[k - 1] = weights[k] * k / (n - k + 1) * (1 - p) / p
    for k in range(mode, n):
        weights[k + 1] = weights[k] * (n - k) / (k + 1) * p / (1 - p)
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def exact_mcnemar_pvalue(native_only, degraded_only):
    """Two-sided conditional exact McNemar; p=1 for no discordances."""
    _validate_n(native_only)
    _validate_n(degraded_only)
    m = native_only + degraded_only
    tail = math.fsum(binomial_pmf(m, 0.5)[:min(native_only, degraded_only) + 1])
    return min(1.0, 2 * tail)


@lru_cache(maxsize=None)
def critical_lower(m, alpha=ALPHA):
    """Reject K<=c or K>=m-c; c=-1 means no possible rejection."""
    _validate_n(m)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    cumulative, critical = 0.0, -1
    for k, probability in enumerate(binomial_pmf(m, 0.5)):
        cumulative += probability
        if 2 * cumulative <= alpha:
            critical = k
        else:
            break
    return critical


def _validate_scenario(q, delta):
    if not math.isfinite(q) or not math.isfinite(delta) or not 0 <= q <= 1 or abs(delta) > q:
        raise ValueError("scenario requires 0<=q<=1 and abs(delta)<=q")


def conditional_rejection(m, p, alpha=ALPHA):
    c = critical_lower(m, alpha)
    if c < 0:
        return 0.0, 0.0
    pmf = binomial_pmf(m, p)
    positive = math.fsum(pmf[m - c:])
    negative = math.fsum(pmf[:c + 1])
    return positive + negative, positive


def _marginalize(n, q, conditional):
    return math.fsum(w * power for w, power in zip(binomial_pmf(n, q), conditional))


def exact_power(n, q, delta, alpha=ALPHA):
    """Unconditional probability of rejection, including a positive-direction view."""
    _validate_n(n)
    _validate_scenario(q, delta)
    critical_lower(0, alpha)  # Validate alpha even when all outcomes agree.
    p = (q + delta) / (2 * q) if q else 0.5
    any_direction, positive = zip(*(conditional_rejection(m, p, alpha) for m in range(n + 1)))
    return {"two_sided_rejection": _marginalize(n, q, any_direction),
            "positive_direction_rejection": _marginalize(n, q, positive)}


def partial_screen_probability(n, q, delta, observed_threshold, alpha=ALPHA):
    """Positive exact rejection AND observed delta cutoff; NOT full advancement."""
    _validate_n(n)
    _validate_scenario(q, delta)
    if not math.isfinite(observed_threshold) or not 0 <= observed_threshold <= 1:
        raise ValueError("observed_threshold must be in [0, 1]")
    cutoff = math.ceil(n * observed_threshold - 1e-12)
    p = (q + delta) / (2 * q) if q else 0.5
    probabilities = []
    for m in range(n + 1):
        c = critical_lower(m, alpha)
        lower = max(m - c, math.ceil((m + cutoff) / 2))
        probabilities.append(0.0 if c < 0 else math.fsum(binomial_pmf(m, p)[lower:]))
    return _marginalize(n, q, probabilities)


def build_plan(feasible_n=62, search_cap=900, reference_n=640, bootstrap_seed=20260927):
    for n in (feasible_n, search_cap, reference_n):
        _validate_n(n)
    if not 0 < feasible_n <= search_cap or not 0 < reference_n <= search_cap:
        raise ValueError("positive feasible_n and reference_n must not exceed search_cap")
    ns = sorted({n for n in (60, feasible_n, 70, reference_n) if n <= search_cap})
    scenarios = []
    z = NormalDist().inv_cdf(0.975)
    for q in DISCORDANCES:
        for delta in EFFECTS:
            p = (q + delta) / (2 * q)
            any_direction, positive = zip(*(conditional_rejection(m, p) for m in range(search_cap + 1)))
            by_n = {str(n): {"two_sided_rejection": _marginalize(n, q, any_direction),
                            "positive_direction_rejection": _marginalize(n, q, positive),
                            "approx_95pct_half_width_pp": 100 * z * math.sqrt((q - delta**2) / n)}
                    for n in ns}
            search = None
            if delta == 0.05:
                curve = [_marginalize(n, q, positive) for n in range(1, search_cap + 1)]
                search = {"criterion": "positive-direction rejection by the two-sided exact test",
                          "max_n_planning_only": search_cap,
                          "minimum_n_for_80pct": next((i + 1 for i, x in enumerate(curve) if x >= 0.8), None),
                          "minimum_n_for_90pct": next((i + 1 for i, x in enumerate(curve) if x >= 0.9), None),
                          "positive_power_at_search_cap": curve[-1],
                          "positive_direction_power_curve": {str(i + 1): value for i, value in enumerate(curve)}}
            scenarios.append({"true_delta": delta, "total_discordance_q": q,
                              "native_only_probability": (q + delta) / 2,
                              "degraded_only_probability": (q - delta) / 2,
                              "power_by_n": by_n, "sample_size_search": search,
                              "partial_screens_at_reference_n_not_full_advancement": {
                                  "n": reference_n,
                                  "positive_exact_test_and_observed_delta_ge_2pp": partial_screen_probability(reference_n, q, delta, 0.02),
                                  "positive_exact_test_and_observed_delta_ge_5pp": partial_screen_probability(reference_n, q, delta, 0.05),
                                  "excludes": ["bootstrap lower bound", "native at least direct", "semantic audit"]}})
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "prospective_planning_only_no_v6_inference_or_model_results",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_metric": "official empty-scale single-span EM",
        "primary_contrast": "native_256 minus degraded_256",
        "metric_applicability": {
            "scope": "Proposed replication of the original TAT-DQA comparison using its TAT-QA-derived single-span scorer",
            "new_dataset_or_scoring_locked": False,
            "new_population_requirement": "For DUDE or another population, prospectively fix the binary primary metric and revisit effect and discordance assumptions; do not silently transfer the TAT-DQA scorer or a claimed power level."},
        "test": {"name": "two-sided exact conditional McNemar/binomial", "alpha": ALPHA,
                 "positive_direction_required_for_benefit": True},
        "feasibility": {"at_most_original_reports_before_semantic_audit": feasible_n,
                        "actual_post_audit_n": None, "fixed_sample_size_locked": False,
                        "source": "Pre-inference strict source and ROI feasibility audit supplied by project lead",
                        "calls_per_report": 4, "at_most_main_calls_before_audit": 4 * feasible_n,
                        "earlier_70_was_inventory_ceiling_not_eligible_count": True},
        "planning_search_ceiling_not_inference_budget": search_cap,
        "theoretical_reference_n_not_available_from_current_pool": reference_n,
        "planning_effect": 0.05, "sensitivity_effects": list(EFFECTS),
        "discordance_scenarios": list(DISCORDANCES),
        "pilot_context_only": {"discordances": 7, "reports": 85, "rate": 7 / 85,
                               "used_to_set_planning_effect_or_n": False},
        "formula": "M~Binomial(N,q); K|M=m~Binomial(m,(q+delta)/(2q)); sum conditional exact rejection over M",
        "assumptions": "Independent eligible original reports; one question/page per report; q>=abs(delta). Company/template dependence may lower effective information.",
        "bootstrap_plan": {"samples": 10000, "seed": bootstrap_seed, "unit": "original_source_report",
                           "interval": "95% percentile paired bootstrap of native minus degraded; all conditions resampled together",
                           "not_an_exact_test_inversion": True},
        "advancement_power": {"full_rule_probability_calculated": False,
                              "reason": "delta and q do not specify joint direct outcomes, bootstrap decision or semantic correctness",
                              "suggested_point_estimate_threshold": 0.02,
                              "planning_effect_is_not_the_observed_gate_threshold": True},
        "rounding_note": "All benefit tables use positive-direction rejection. Earlier any-direction N70 powers at delta=.05 rounded to 11.3% and 9.8% for q=.15/.20; positive-direction values round to 11.2% and 9.7%. These are different quantities, not a computational disagreement.",
        "scenarios": scenarios,
    }


def render_report(plan):
    n = plan["feasibility"]["at_most_original_reports_before_semantic_audit"]
    ref = plan["theoretical_reference_n_not_available_from_current_pool"]
    cap = plan["planning_search_ceiling_not_inference_budget"]
    rows, searches, partial_five = [], [], []
    for item in plan["scenarios"]:
        x = item["power_by_n"][str(n)]
        rows.append(f"| {item['total_discordance_q']:.2f} | {100 * item['true_delta']:g} | {100 * x['two_sided_rejection']:.2f}% | {100 * x['positive_direction_rejection']:.2f}% | {x['approx_95pct_half_width_pp']:.2f} |")
        s = item["sample_size_search"]
        if s:
            n80, n90 = (s[key] or f">{cap}" for key in ("minimum_n_for_80pct", "minimum_n_for_90pct"))
            searches.append(f"| {item['total_discordance_q']:.2f} | {n80} | {n90} | {100 * item['power_by_n'][str(ref)]['positive_direction_rejection']:.2f}% |")
            partial_five.append(item["partial_screens_at_reference_n_not_full_advancement"]["positive_exact_test_and_observed_delta_ge_5pp"])
    return f"""# Prospective Qwen replication: power and precision

**Planning only; no v6 model outcomes or inference.** The strict source/ROI feasibility audit permits **at most {n} new original reports before semantic review**, or {4*n} main model calls. Final eligible N is not yet locked. The earlier 70-report inventory bound was before stricter eligibility. Neither the reference N={ref} nor the numerical search through N={cap} is an inference budget or an available sample from the current pool.

The proposed original TAT-DQA replication has four conditions per report: direct256, fresh native256, matched overview-derived degraded256, and highres4096. Its proposed sole primary outcome is the TAT-QA-derived official empty-scale single-span EM; the primary contrast is native256 minus degraded256, preserving v5 prompts and geometry. **No new dataset or scoring rule is locked by this planning artifact.** For DUDE or another population, fix the binary primary metric separately before inference and revisit effect/discordance assumptions; the TAT-DQA scorer and these power assumptions cannot be transferred silently. ROIs remain annotation-derived privileged positive controls, so even replication would not validate a deployable selector. Other metrics and conditions are secondary.

## Assumptions and exact calculation

Let p10=P(native correct, degraded wrong), p01 the reverse, delta=p10-p01 and q=p10+p01. The planning alternative is **+5 percentage points versus zero**, with +3/+3.5 pp sensitivity and q=.08/.15/.20. These are scenarios rather than estimates guaranteed on new reports. The known pilot 7/85 discordance rate is context only; its observed net gain did not select the effect or N. One question/page per original report is the unit. Residual company/template dependence may reduce effective information.

The primary test is two-sided exact McNemar at alpha=.05, with positive observed direction required for benefit. Given M discordances, its null is Binomial(M,.5); p=min(1,2*P[K<=min(n10,n01)]), with p=1 for M=0. See the primary documentation for [statsmodels McNemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html) and [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html). Do not replace this test with mid-p or an asymptotic test after outcomes.

The standard-library calculation enumerates M~Binomial(N,q) and K|M~Binomial(M,(q+delta)/(2q)). It is deterministic numerical enumeration, not simulated power. Both any-direction rejection and positive-direction rejection are retained. The latter is the chance of detecting benefit with the two-sided test.

## Attainable N={n}, before semantic exclusions

| Discordance q | True gain, pp | Any-direction rejection | Benefit rejection | Approx. 95% half-width, pp |
|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The half-width uses only 1.96*sqrt((q-delta^2)/N): it is a planning approximation, **not an exact or bootstrap interval and not a guaranteed width**. Small-count discreteness is material: five native-only wins and no reverse loss give p=.0625. This available panel cannot provide 80% power for the planned modest effect. Failure to reject would be inconclusive, not evidence of equivalence or no useful gain. Semantic exclusions can reduce N further; freeze and recalculate for the actual eligible N before inference.

## Sample size needed under +5 pp scenarios

Minimum integer N first reaching the stated probability of **positive-direction rejection by the two-sided exact test**, searched up to {cap}. This is planning only. The rows are different assumed discordance regimes, not alternative panels to select after results.

| Discordance q | N for 80% | N for 90% | Benefit power at theoretical N={ref} |
|---:|---:|---:|---:|
{chr(10).join(searches)}

No complete sample-size search is claimed for +3/+3.5 pp; their attainable-panel and reference-N sensitivity values are in JSON. Securing another dataset changes population scope and requires a new pre-inference selection decision. More pages or questions from the same report do not manufacture independent reports; that would require a different clustered estimand and justified within-report dependence assumptions.

## Statistical evidence and advancement differ

Report paired delta, n10/n01 and exact p. Also report a 95% percentile paired bootstrap CI from 10,000 original-report resamples, seed {plan['bootstrap_plan']['seed']}; carry all four condition outcomes together. This CI is not the inversion of exact McNemar, and boundary disagreements can occur. Report both. A conditional win-probability interval for n10/M is not an interval for delta=(n10-n01)/N. Near-zero observed variance is not equivalence evidence.

A prospective advancement rule could require observed delta>=2 pp, positive direction, exact p<=.05, bootstrap lower bound>0, native EM>=direct EM, and a separately fixed semantic screen. The 5 pp alternative is for planning, not the observed-gain cutoff. A 2 pp point estimate plus CI lower>0 does not prove that the true effect exceeds 2 pp. No full advancement probability has been calculated.

JSON's **partial screens** combine only positive exact rejection and an observed-gain cutoff; they exclude the CI, direct comparison and semantic audit. At true +5 pp and theoretical N={ref}, requiring observed delta>=5 pp yields {100*min(partial_five):.1f}% to {100*max(partial_five):.1f}% partial-screen passage across these scenarios. Full advancement can be no more likely and needs additional joint outcome assumptions. The retained 2 pp cutoff also does not guarantee any target full-rule power.

## Freeze, interpretation and reproducibility

Bind actual N, all report identities, source-only eligibility, model/prompts, geometry, metric, exact test, interval and advancement rule before inference. Invalid/truncated responses stay in denominators; all planned paired records are required. No interim quality inspection, optional extension after results, replacement of difficult cases, or pooling earlier development results into the new estimate. If only this small panel is available, explicitly label it a low-power check on new reports or pause for an adequate untouched pool.

Rounding reconciliation: earlier N70 messages quoting 11.3%/9.8% at delta=5 pp, q=.15/.20 used any-direction rejection; benefit rejection is 11.2%/9.7%. Both values are retained in JSON. The table above separates them and uses the corrected current feasibility ceiling.

Run from the project: `python experiments/plan_replication_power.py --output reports/replication_plan`. Optional parameters are `--feasible-n`, `--search-cap`, `--reference-n`, and `--bootstrap-seed`. The script reads no experiment records and writes only the selected planning output directory. Numerical unit tests independently enumerate multinomial paired outcomes and exact integer binomial rejection regions, check probability moments/null size, and test the partial-screen distinction. No existing frozen study files are changed.

The JSON includes all integer N=1 through {cap} for +5 pp under `sample_size_search.positive_direction_power_curve`; thresholds use that same positive-direction criterion. SciPy was unavailable in both existing Python environments checked for this study, so no SciPy agreement is claimed. The independent exact rational/integer tests require no SciPy dependency.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Planning output directory for power_plan.json and power_plan.md")
    parser.add_argument("--feasible-n", type=int, default=62)
    parser.add_argument("--search-cap", type=int, default=900, help="Numerical planning search only, not an inference budget")
    parser.add_argument("--reference-n", type=int, default=640)
    parser.add_argument("--bootstrap-seed", type=int, default=20260927)
    args = parser.parse_args()
    plan = build_plan(args.feasible_n, args.search_cap, args.reference_n, args.bootstrap_seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "power_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "power_plan.md").write_text(render_report(plan), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": plan["status"], "feasible_n_before_audit": args.feasible_n}))


if __name__ == "__main__":
    main()
