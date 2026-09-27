"""Independent numerical checks for prospective paired-binary power."""
from fractions import Fraction
import importlib.util
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "plan_replication_power", Path(__file__).resolve().parents[1] / "experiments" / "plan_replication_power.py")
power = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(power)


def integer_pvalue(wins, losses):
    m = wins + losses
    return min(Fraction(1), Fraction(2 * sum(math.comb(m, j) for j in range(min(wins, losses) + 1)), 2**m))


def multinomial_reference(n, q, delta, threshold=None):
    """Sum paired outcome counts directly, without M/K decomposition or PMFs."""
    plus, minus, tie = (q + delta) / 2, (q - delta) / 2, 1 - q
    positive, total = Fraction(0), Fraction(0)
    for wins in range(n + 1):
        for losses in range(n - wins + 1):
            if integer_pvalue(wins, losses) > Fraction(1, 20):
                continue
            count = math.comb(n, wins) * math.comb(n - wins, losses)
            probability = count * plus**wins * minus**losses * tie**(n - wins - losses)
            total += probability
            if wins > losses and (threshold is None or Fraction(wins - losses, n) >= threshold):
                positive += probability
    return float(total), float(positive)


@pytest.mark.parametrize("p", [0, 0.0001, 0.04, 0.5, 0.8125, 0.9999, 1])
def test_binomial_distribution_moments_at_long_search_range(p):
    n = 900
    pmf = power.binomial_pmf(n, p)
    assert math.fsum(pmf) == pytest.approx(1, abs=1e-14)
    assert math.fsum(k * x for k, x in enumerate(pmf)) == pytest.approx(n * p, abs=1e-10)
    assert math.fsum((k - n*p)**2 * x for k, x in enumerate(pmf)) == pytest.approx(n*p*(1-p), abs=1e-9)


def test_exact_rejection_regions_match_integer_binomial_coefficients():
    for m in range(61):
        c = power.critical_lower(m)
        for wins in range(m + 1):
            expected = integer_pvalue(wins, m - wins)
            assert power.exact_mcnemar_pvalue(wins, m - wins) == pytest.approx(float(expected), abs=1e-14)
            rejected = c >= 0 and (wins <= c or wins >= m - c)
            assert rejected == (expected <= Fraction(1, 20))
    assert power.exact_mcnemar_pvalue(5, 0) == pytest.approx(0.0625, abs=1e-15)
    assert power.exact_mcnemar_pvalue(6, 0) == pytest.approx(0.03125, abs=1e-15)
    assert power.exact_mcnemar_pvalue(0, 0) == 1


@pytest.mark.parametrize("n,q,delta", [(6, Fraction(1, 5), Fraction(1, 20)),
    (12, Fraction(3, 5), Fraction(1, 5)), (25, Fraction(1, 5), Fraction(1, 20)),
    (25, Fraction(1), Fraction(1, 5))])
def test_unconditional_power_matches_independent_multinomial_enumeration(n, q, delta):
    any_direction, positive = multinomial_reference(n, q, delta)
    result = power.exact_power(n, float(q), float(delta))
    assert result["two_sided_rejection"] == pytest.approx(any_direction, abs=1e-13)
    assert result["positive_direction_rejection"] == pytest.approx(positive, abs=1e-13)
    if n == 6:
        assert positive == pytest.approx(float(((q + delta) / 2)**6), abs=1e-15)


def test_partial_screen_is_only_rejection_and_point_cutoff():
    n, q, delta, threshold = 25, Fraction(3, 5), Fraction(1, 5), Fraction(2, 5)
    _, expected = multinomial_reference(n, q, delta, threshold)
    actual = power.partial_screen_probability(n, float(q), float(delta), float(threshold))
    assert actual == pytest.approx(expected, abs=1e-13)
    assert actual < power.exact_power(n, float(q), float(delta))["positive_direction_rejection"]


@pytest.mark.parametrize("q", [0.08, 0.15, 0.20])
def test_null_size_and_positive_direction_are_distinct(q):
    null = power.exact_power(62, q, 0)
    assert null["two_sided_rejection"] <= 0.05
    assert null["positive_direction_rejection"] * 2 == pytest.approx(null["two_sided_rejection"], abs=1e-14)
    positive, negative = power.exact_power(62, q, 0.05), power.exact_power(62, q, -0.05)
    assert positive["two_sided_rejection"] == pytest.approx(negative["two_sided_rejection"], abs=1e-13)
    assert positive["positive_direction_rejection"] > negative["positive_direction_rejection"]
    assert positive["positive_direction_rejection"] < positive["two_sided_rejection"]


def test_no_discordances_or_insufficient_pairs_never_reject():
    assert power.exact_power(62, 0, 0) == {"two_sided_rejection": 0, "positive_direction_rejection": 0}
    assert power.exact_power(5, 1, 1)["two_sided_rejection"] == 0
    assert power.exact_power(6, 1, 1)["positive_direction_rejection"] == 1


@pytest.mark.parametrize("n,q,delta", [(-1, .1, 0), (1.5, .1, 0), (5, .1, .11),
    (5, -.1, 0), (5, 1.1, 0), (5, .1, float("nan"))])
def test_impossible_designs_are_rejected(n, q, delta):
    with pytest.raises(ValueError):
        power.exact_power(n, q, delta)


def test_plan_separates_available_cohort_search_range_and_partial_screens():
    plan = power.build_plan(feasible_n=62, search_cap=64, reference_n=62)
    assert plan["feasibility"]["at_most_main_calls_before_audit"] == 248
    assert plan["feasibility"]["actual_post_audit_n"] is None
    assert plan["planning_search_ceiling_not_inference_budget"] == 64
    assert plan["bootstrap_plan"]["seed"] == 20260927
    assert plan["metric_applicability"]["new_dataset_or_scoring_locked"] is False
    assert plan["advancement_power"]["full_rule_probability_calculated"] is False
    assert len(plan["scenarios"]) == 9
    for row in plan["scenarios"]:
        if row["true_delta"] == 0.05:
            search = row["sample_size_search"]
            assert list(search["positive_direction_power_curve"]) == [str(n) for n in range(1, 65)]
            assert search["positive_direction_power_curve"]["62"] == pytest.approx(row["power_by_n"]["62"]["positive_direction_rejection"])
            assert search["minimum_n_for_80pct"] is None
        else:
            assert row["sample_size_search"] is None
        assert "bootstrap lower bound" in row["partial_screens_at_reference_n_not_full_advancement"]["excludes"]
    report = power.render_report(plan)
    assert "at most 62 new original reports before semantic review" in report
    assert "No full advancement probability has been calculated" in report
