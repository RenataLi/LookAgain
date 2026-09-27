"""Verify aggregate arithmetic from precomputed, text-free numeric observations.

This does not score model answers, inspect documents, fit rankers, or revalidate
the original execution. Only the standard library is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("primary_em", "official_anls")
POLICIES = ("ranker_full", "ranker_image_position", "ranker_position", "best_fixed",
            "center", "random", "prompted", "direct", "highres", "uniform", "oracle9")
STRATEGIES = tuple(p + "_" + v for p in POLICIES for v in ("new", "old"))
COSTS = ("elapsed_s", "input_tokens", "visual_tokens", "generated_tokens",
         "peak_memory_gib", "peak_reserved_gib", "processor_observer_elapsed_s",
         "cpu_policy_elapsed_s")
CONTRASTS = tuple(p for p in POLICIES if p not in ("prompted", "oracle9"))


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    def invalid(_):
        raise ValueError("Nonfinite JSON number")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), label + ": schema differs")


def number(value, low=0., high=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def validate(data):
    keys(data, ("schema_version", "n_sources", "fold_assignments", "bootstrap_samples",
                "bootstrap_seed", "strategies", "uniform_new_marginal_cost_observations"), "scores")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1, "Wrong schema version")
    n = data["n_sources"]
    require(type(n) is int and n >= 2, "Invalid source count")
    folds = data["fold_assignments"]
    require(isinstance(folds, list) and len(folds) == n and all(type(f) is int and f >= 0 for f in folds), "Invalid folds")
    require(2 <= len(set(folds)) <= n and set(folds) == set(range(len(set(folds)))), "Missing fold")
    require(type(data["bootstrap_samples"]) is int and 1 <= data["bootstrap_samples"] <= 100000, "Invalid bootstrap count")
    require(type(data["bootstrap_seed"]) is int, "Invalid bootstrap seed")
    keys(data["strategies"], STRATEGIES, "strategies")

    def costs(rows, length):
        require(isinstance(rows, list) and len(rows) == length, "Cost observation count differs")
        for row in rows:
            keys(row, COSTS, "cost observation")
            require(all(number(row[k]) for k in COSTS), "Invalid/nonfinite cost")
            require(row["peak_reserved_gib"] >= row["peak_memory_gib"], "Reserved memory below allocated")

    for name, policy in data["strategies"].items():
        keys(policy, ("scores", "cost_observations", "historical_answer_cost_observations", "invalid", "truncated"), "policy")
        keys(policy["scores"], METRICS, "policy scores")
        for metric, values in policy["scores"].items():
            require(isinstance(values, list) and len(values) == n and all(number(x, 0, 1) for x in values), "Invalid/nonfinite scores")
            if metric == "primary_em" and not name.startswith("uniform_"):
                require(all(x in (0, 1) for x in values), "Nonbinary EM outside uniform expectation")
        oracle = name.startswith("oracle9_")
        for flag in ("invalid", "truncated"):
            values = policy[flag]
            require(isinstance(values, list) and len(values) == n, "Flag observation count differs")
            if oracle:
                require(all(x is None for x in values), "Oracle flags must be unavailable")
            else:
                require(all(type(x) is bool or number(x, 0, 1) for x in values), "Invalid/nonfinite flag")
                if not name.startswith("uniform_"):
                    require(all(x in (0, 1) for x in values), "Nonbinary flag outside uniform expectation")
        costs(policy["cost_observations"], n if name.endswith("_new") and not oracle else 0)
        costs(policy["historical_answer_cost_observations"], n if name.endswith("_old") and not oracle else 0)
    costs(data["uniform_new_marginal_cost_observations"], n * 9)
    return n, folds


def mean(values):
    return math.fsum(values) / len(values)


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def describe(values):
    return {"n": len(values), "mean": mean(values), "median": quantile(values, .5),
            "p95": quantile(values, .95), "min": min(values), "max": max(values)}


def describe_cost(rows):
    return {key: describe([r[key] for r in rows]) for key in COSTS} if rows else None


def paired(a, b, folds):
    delta = [x - y for x, y in zip(a, b)]
    result = {"n_sources": len(a), "mean_difference": mean(delta), "per_source_differences": delta,
              "folds": [{"fold": f, "n_sources": folds.count(f),
                         "a_mean": mean([x for x, t in zip(a, folds) if t == f]),
                         "b_mean": mean([x for x, t in zip(b, folds) if t == f]),
                         "mean_difference": mean([x for x, t in zip(delta, folds) if t == f])}
                        for f in sorted(set(folds))]}
    return result


def cells(a, b):
    return {"repairs": sum(x == 1 and y == 0 for x, y in zip(a, b)),
            "harms": sum(x == 0 and y == 1 for x, y in zip(a, b)),
            "both_correct": sum(x == 1 and y == 1 for x, y in zip(a, b)),
            "both_wrong": sum(x == 0 and y == 0 for x, y in zip(a, b))}


def compute(data):
    n, folds = validate(data)
    result = {"schema_version": 1, "n_sources": n, "strategies": {},
              "descriptive_policy_contrasts": {}, "factorial_interaction": {}, "fixed_policy_prompt_effect": {}}
    for name, row in data["strategies"].items():
        entry = {"metrics": {m: {"mean": mean(row["scores"][m]), "n_sources": n} for m in METRICS},
                 "cost": describe_cost(row["cost_observations"]),
                 "historical_answer_cost": describe_cost(row["historical_answer_cost_observations"]),
                 "deployed_policy": not name.startswith("oracle9_")}
        for flag in ("invalid", "truncated"):
            entry[flag + "_count_or_expected_count"] = (None if any(x is None for x in row[flag])
                                                       else math.fsum(float(x) for x in row[flag]))
        if name == "uniform_new":
            entry["marginal_single_crop_cost"] = describe_cost(data["uniform_new_marginal_cost_observations"])
        result["strategies"][name] = entry
    score = lambda name, metric: data["strategies"][name]["scores"][metric]
    for policy in CONTRASTS:
        name = policy + "_new_minus_prompted_new"
        result["descriptive_policy_contrasts"][name] = {}
        for metric in METRICS:
            a, b = score(policy + "_new", metric), score("prompted_new", metric)
            value = paired(a, b, folds)
            if metric == "primary_em" and all(x in (0, 1) for x in a):
                value.update(cells(a, b))
            result["descriptive_policy_contrasts"][name][metric] = value
    result["primary"] = result["descriptive_policy_contrasts"]["ranker_full_new_minus_prompted_new"]
    for metric in METRICS:
        new = [x-y for x, y in zip(score("ranker_full_new", metric), score("prompted_new", metric))]
        old = [x-y for x, y in zip(score("ranker_full_old", metric), score("prompted_old", metric))]
        result["factorial_interaction"][metric] = paired(new, old, folds)
        a, b = score("prompted_new", metric), score("prompted_old", metric)
        delta = [x-y for x, y in zip(a, b)]
        rng = random.Random(data["bootstrap_seed"])
        means = [mean([delta[rng.randrange(n)] for _ in range(n)]) for _ in range(data["bootstrap_samples"])]
        value = {"n_sources": n, "mean_difference": mean(delta), "per_source_differences": delta,
                 "development_paired_bootstrap_ci95": [quantile(means, .025), quantile(means, .975)]}
        if metric == "primary_em":
            value.update(cells(a, b))
        result["fixed_policy_prompt_effect"][metric] = value
    return result


def compare(actual, expected, label="summary"):
    """Count checked scalar values. Exact structure, tight numeric tolerance."""
    if isinstance(expected, dict):
        keys(actual, expected, label)
        return sum(compare(actual[k], v, label + "." + k) for k, v in expected.items())
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ": length differs")
        return sum(compare(a, b, label + "[]") for a, b in zip(actual, expected))
    if type(expected) in (int, float):
        require(number(expected, -math.inf) and number(actual, -math.inf)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), label + ": number differs")
    else:
        require((expected is None or type(expected) is bool) and type(actual) is type(expected)
                and actual == expected, label + ": invalid scalar")
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "reports/benchmark/scores.json")
    parser.add_argument("--expected", type=Path, default=ROOT / "reports/benchmark/expected_summary.json")
    parser.add_argument("--output", type=Path, help="New output directory; never overwritten")
    args = parser.parse_args()
    output = args.output or ROOT / "generated" / ("numeric-replay-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    require(not output.exists(), "Output directory already exists")
    actual = compute(load(args.scores))
    count = compare(actual, load(args.expected))
    receipt = {"status": "PASS", "scalar_comparisons": count, "n_sources": actual["n_sources"],
               "n_strategies": len(actual["strategies"]), "new_model_calls": 0,
               "raw_answer_rescoring": False, "source_identity_verification": False,
               "model_or_feature_verification": False, "ranker_refitted": False,
               "primary_difference_pp": 100 * actual["primary"]["primary_em"]["mean_difference"]}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary.json", actual), ("verification.json", receipt)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PASS: {count:,} scalar arithmetic comparisons; {actual['n_sources']} aligned numeric rows; {len(actual['strategies'])} strategies.")
    print("This verifies precomputed-score arithmetic, not raw answers, source identity, features or training.")
    print("Reports: " + str(output))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print("Verification failed: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
