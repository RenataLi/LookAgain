"""Arithmetic and fail-closed schema tests; no document text or model calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "numeric_replay", Path(__file__).resolve().parents[1] / "scripts/reproduce_results.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


@pytest.fixture
def panel():
    n = 5
    costs = [{k: (float(i + 1) if k == "elapsed_s" else 10.) for k in replay.COSTS} for i in range(n)]
    data = {"schema_version": 1, "n_sources": n, "fold_assignments": [0, 1, 0, 1, 0],
            "bootstrap_samples": 20, "bootstrap_seed": 23, "strategies": {},
            "uniform_new_marginal_cost_observations": deepcopy(costs * 9)}
    for name in replay.STRATEGIES:
        oracle = name.startswith("oracle9_")
        grades = [1, 0, 1, 0, 1]
        if name == "prompted_new":
            grades = [0, 0, 1, 1, 0]
        if name == "prompted_old":
            grades = [0, 1, 1, 0, 0]
        data["strategies"][name] = {
            "scores": {m: list(grades) for m in replay.METRICS},
            "cost_observations": deepcopy(costs) if name.endswith("_new") and not oracle else [],
            "historical_answer_cost_observations": deepcopy(costs) if name.endswith("_old") and not oracle else [],
            "invalid": [None if oracle else False] * n,
            "truncated": [None if oracle else False] * n}
    return data


def test_known_paired_effect_fold_weighting_and_costs(panel):
    result = replay.compute(panel)
    primary = result["primary"]["primary_em"]
    assert primary["mean_difference"] == pytest.approx(.2)
    assert (primary["repairs"], primary["harms"], primary["both_correct"], primary["both_wrong"]) == (2, 1, 1, 1)
    assert primary["folds"][0]["mean_difference"] == pytest.approx(2 / 3)
    assert primary["folds"][1]["mean_difference"] == -.5
    cost = result["strategies"]["ranker_full_new"]["cost"]["elapsed_s"]
    assert cost == {"n": 5, "mean": 3., "median": 3., "p95": 4.8, "min": 1., "max": 5.}
    assert result["strategies"]["oracle9_new"]["cost"] is None


@pytest.mark.parametrize("field", ["score", "cost", "fold", "flag"])
def test_tampered_observation_changes_verified_arithmetic(panel, field):
    expected = replay.compute(panel)
    row = panel["strategies"]["ranker_full_new"]
    if field == "score":
        row["scores"]["primary_em"][0] = 0
    elif field == "cost":
        row["cost_observations"][0]["elapsed_s"] += .2
    elif field == "fold":
        panel["fold_assignments"][0] = 1
    else:
        row["invalid"][0] = True
    with pytest.raises(ValueError, match="differs"):
        replay.compare(replay.compute(panel), expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_invalid_numeric_score_rejected(panel, value):
    panel["strategies"]["direct_new"]["scores"]["primary_em"][0] = value
    with pytest.raises(ValueError):
        replay.compute(panel)


def test_nonfinite_cost_rejected(panel):
    panel["strategies"]["direct_new"]["cost_observations"][0]["elapsed_s"] = float("nan")
    with pytest.raises(ValueError):
        replay.compute(panel)


def test_unknown_fields_cannot_smuggle_document_text(panel):
    panel["question"] = "synthetic unexpected text"
    with pytest.raises(ValueError, match="schema differs"):
        replay.compute(panel)


def test_missing_strategy_or_row_cannot_shrink_denominator(panel):
    del panel["strategies"]["direct_old"]
    with pytest.raises(ValueError, match="schema differs"):
        replay.compute(panel)


def test_short_cost_panel_rejected(panel):
    panel["strategies"]["direct_new"]["cost_observations"].pop()
    with pytest.raises(ValueError, match="count differs"):
        replay.compute(panel)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_json_nonfinite_literals_rejected(tmp_path, literal):
    path = tmp_path / "numeric.json"
    path.write_text('{"n":' + literal + '}', encoding="utf-8")
    with pytest.raises(ValueError, match="Nonfinite"):
        replay.load(path)


def test_expected_summary_tamper_rejected(panel):
    actual = replay.compute(panel)
    expected = deepcopy(actual)
    expected["primary"]["primary_em"]["repairs"] += 1
    with pytest.raises(ValueError):
        replay.compare(actual, expected)


def test_published_numeric_artifacts_have_no_string_values_and_replay():
    root = Path(__file__).resolve().parents[1]
    data = replay.load(root / "reports/benchmark/scores.json")
    expected = replay.load(root / "reports/benchmark/expected_summary.json")

    def numeric_leaves(value):
        if isinstance(value, dict):
            return all(numeric_leaves(x) for x in value.values())
        if isinstance(value, list):
            return all(numeric_leaves(x) for x in value)
        return value is None or type(value) in (int, float, bool)

    assert numeric_leaves(data) and numeric_leaves(expected)
    count = replay.compare(replay.compute(data), expected)
    assert count == 4806
    assert expected["n_sources"] == 124 and len(expected["strategies"]) == 22
