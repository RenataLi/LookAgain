"""Paired, image-clustered analysis of one-step LookAgain experiments.

Every follow-up record stores *incremental* elapsed time. Its policy cost is
therefore ``direct.elapsed_s + action.elapsed_s``. ``highres`` is a separate,
standalone policy. Selection/proposal overhead is included only if the caller
already included it in the recorded timing.

Examples with a missing, duplicate, failed, or malformed declared action are
excluded from *every* comparison and are explicitly described in coverage.
The resulting complete-case estimates do not correct for nonrandom failures.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_ACTIONS = (
    "direct", "highres", "recheck", "think",
    "crop_tl", "crop_tr", "crop_bl", "crop_br",
)


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a quantile of an empty sequence.")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _record_problem(record: dict) -> str | None:
    if record.get("status") != "ok":
        return f"status={record.get('status', 'missing')}"
    if not isinstance(record.get("correct"), bool):
        return "correct must be a boolean"
    for field in ("elapsed_s", "peak_memory_gib"):
        if not _nonnegative_number(record.get(field)):
            return f"{field} must be a finite nonnegative number"
    return None


def _policy_observation(group: dict[str, dict], action: str) -> dict:
    record = group[action]
    direct = group["direct"]
    standalone = action in ("direct", "highres")
    return {
        "correct": record["correct"],
        "total_elapsed_s": float(record["elapsed_s"]) + (
            0.0 if standalone else float(direct["elapsed_s"])
        ),
        # Sequential allocations are not added together.
        "peak_memory_gib": float(record["peak_memory_gib"]) if standalone else
        max(float(record["peak_memory_gib"]), float(direct["peak_memory_gib"])),
        "selected_action": action,
    }


def _metrics(observations: list[dict], direct: list[dict]) -> dict:
    count = len(observations)
    accuracy = sum(row["correct"] for row in observations) / count
    base_accuracy = sum(row["correct"] for row in direct) / count
    fixes = sum(not base["correct"] and row["correct"]
                for base, row in zip(direct, observations))
    harms = sum(base["correct"] and not row["correct"]
                for base, row in zip(direct, observations))
    latencies = [row["total_elapsed_s"] for row in observations]
    memories = [row["peak_memory_gib"] for row in observations]
    return {
        "n_examples": count,
        "accuracy": accuracy,
        "delta_accuracy_pp": 100.0 * (accuracy - base_accuracy),
        "delta_ci_95_pp": None,
        "wrong_to_right_count": fixes,
        "right_to_wrong_count": harms,
        "wrong_to_right_fraction": fixes / count,
        "right_to_wrong_fraction": harms / count,
        "total_latency_s": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": _quantile(latencies, 0.95),
        },
        "policy_peak_memory_gib": {
            "mean": statistics.mean(memories), "max": max(memories),
        },
    }


def analyze_records(
    records: list[dict],
    bootstrap_samples: int = 2000,
    seed: int = 20260925,
    expected_actions: Iterable[str] | None = None,
) -> dict:
    """Return JSON-compatible results, including explicit exclusion coverage.

    ``expected_actions`` must be specified from the experiment configuration,
    never inferred from whichever actions happened to succeed. It defaults to
    all eight supported actions and must contain ``direct``. Extra actions in
    the input are counted as ignored. Duplicate declared actions invalidate an
    example; this function does not guess which retry should be retained.

    The percentile bootstrap resamples entire image clusters, preserving all
    questions and action pairs within each sampled image. The point estimate
    remains example-weighted. Fewer than two complete image clusters produce
    no confidence intervals. No complete groups produce coverage but no metrics.

    Random-crop choices are drawn once, uniformly, using the fixed seed and
    sorted example IDs. They are held fixed during the image bootstrap. The
    hindsight oracle is privileged and diagnostic, not a deployable method.
    """
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer for reproducible analysis.")
    if not isinstance(records, list):
        raise TypeError("records must be a list of dictionaries.")
    actions = tuple(DEFAULT_ACTIONS if expected_actions is None else expected_actions)
    if not actions or "direct" not in actions or len(set(actions)) != len(actions):
        raise ValueError("expected_actions must be unique and include direct.")
    if set(actions) - set(DEFAULT_ACTIONS):
        raise ValueError("expected_actions contains unsupported actions.")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a dictionary.")
        for field in ("example_id", "image_id", "action"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"Record {index} needs a nonempty string {field}.")
        grouped[record["example_id"]].append(record)

    per_action = {action: {
        "ok_records": 0, "failed_records": 0, "invalid_records": 0,
        "missing_examples": 0, "duplicate_examples": 0,
    } for action in actions}
    ignored = Counter()
    excluded = []
    complete: list[tuple[str, str, dict[str, dict]]] = []
    for example_id in sorted(grouped):
        rows = grouped[example_id]
        image_ids = sorted({row["image_id"] for row in rows})
        by_action: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            action = row["action"]
            if action not in per_action:
                ignored[action] += 1
                continue
            by_action[action].append(row)
            if row.get("status") != "ok":
                per_action[action]["failed_records"] += 1
            elif _record_problem(row):
                per_action[action]["invalid_records"] += 1
            else:
                per_action[action]["ok_records"] += 1
        problems = {}
        if len(image_ids) != 1:
            problems["image_id"] = "inconsistent image_id within example"
        for action in actions:
            candidates = by_action[action]
            if not candidates:
                per_action[action]["missing_examples"] += 1
                problems[action] = "missing"
            elif len(candidates) != 1:
                per_action[action]["duplicate_examples"] += 1
                problems[action] = f"duplicate: {len(candidates)} records"
            else:
                problem = _record_problem(candidates[0])
                if problem:
                    problems[action] = problem
        if problems:
            excluded.append({"example_id": example_id, "image_ids": image_ids, "problems": problems})
        else:
            complete.append((example_id, image_ids[0], {action: by_action[action][0] for action in actions}))

    warnings = []
    if excluded:
        warnings.append(
            "Incomplete coverage: all comparisons use the same complete examples only. "
            "Failures may be nonrandom; these estimates do not establish performance on the full intended set."
        )
    if ignored:
        warnings.append("Undeclared action records were ignored; inspect coverage.ignored_action_records.")
    image_ids = sorted({item[1] for item in complete})
    summary = {
        "schema_version": 1,
        "status": "complete" if complete and not excluded else "incomplete_coverage" if complete else "no_complete_groups",
        "expected_actions": list(actions),
        "coverage": {
            "records_seen": len(records), "example_groups_seen": len(grouped),
            "complete_examples": len(complete), "complete_images": len(image_ids),
            "excluded_examples": len(excluded),
            "completion_fraction": len(complete) / len(grouped) if grouped else 0.0,
            "per_action": per_action, "excluded_groups": excluded,
            "ignored_action_records": dict(sorted(ignored.items())),
            "scope": "Coverage is relative to examples present in records; entirely absent examples require a manifest check.",
        },
        "bootstrap": {
            "samples": bootstrap_samples, "seed": seed, "unit": "image_id",
            "interval": "95% percentile", "estimator": "example-weighted paired accuracy difference",
        },
        "cost_accounting": {
            "standalone_actions": [action for action in actions if action in ("direct", "highres")],
            "followup_latency": "direct.elapsed_s + followup.elapsed_s",
            "followup_peak_memory": "max(direct.peak_memory_gib, followup.peak_memory_gib)",
            "limitation": "Selection/proposal overhead is included only when already recorded in elapsed_s.",
        },
        "actions": {}, "baselines": {}, "warnings": warnings,
    }
    if not complete:
        warnings.append("No complete paired groups; accuracy and cost comparisons are unavailable.")
        return summary

    observations = {
        action: [_policy_observation(group, action) for _, _, group in complete]
        for action in actions
    }
    direct = observations["direct"]
    for action in actions:
        summary["actions"][action] = _metrics(observations[action], direct)

    crop_actions = sorted(action for action in actions if action.startswith("crop_"))
    if crop_actions:
        selection_rng = random.Random(seed)
        random_rows = [
            _policy_observation(group, selection_rng.choice(crop_actions))
            for _, _, group in complete
        ]
        observations["uniform_random_crop"] = random_rows
        summary["baselines"]["uniform_random_crop"] = {
            **_metrics(random_rows, direct),
            "label": "Uniform random crop (one fixed seeded draw per example)",
            "selection_seed": seed,
            "candidate_actions": crop_actions,
            "selected_action_counts": dict(sorted(Counter(row["selected_action"] for row in random_rows).items())),
        }

    oracle_actions = ["direct"] + sorted(action for action in actions if action not in ("direct", "highres"))
    oracle_rows = []
    for _, _, group in complete:
        correct_candidates = [_policy_observation(group, action) for action in oracle_actions if group[action]["correct"]]
        # Stable order selects direct on tied costs. If every action is wrong,
        # stop instead of pretending that a failed extra action had utility.
        chosen = min(correct_candidates, key=lambda row: row["total_elapsed_s"]) if correct_candidates else _policy_observation(group, "direct")
        oracle_rows.append(chosen)
    observations["hindsight_oracle"] = oracle_rows
    summary["baselines"]["hindsight_oracle"] = {
        **_metrics(oracle_rows, direct),
        "label": "Hindsight oracle (privileged diagnostic; not deployable)",
        "candidate_actions": oracle_actions,
        "selection_rule": "Minimum total latency among correct candidates; choose direct if none is correct.",
        "limitation": "Uses ground-truth correctness for selection and charges only the selected candidate; excludes highres.",
        "selected_action_counts": dict(sorted(Counter(row["selected_action"] for row in oracle_rows).items())),
    }

    if len(image_ids) < 2:
        warnings.append("Fewer than two complete image clusters; confidence intervals are unavailable.")
        return summary

    image_positions: dict[str, list[int]] = defaultdict(list)
    for position, (_, image_id, _) in enumerate(complete):
        image_positions[image_id].append(position)
    method_names = list(observations)
    cluster_sizes = [len(image_positions[image_id]) for image_id in image_ids]
    cluster_deltas = {
        method: [sum(int(observations[method][i]["correct"]) - int(direct[i]["correct"])
                     for i in image_positions[image_id]) for image_id in image_ids]
        for method in method_names
    }
    bootstrap_rng = random.Random(seed)
    draws = {method: [] for method in method_names}
    for _ in range(bootstrap_samples):
        counts = Counter(bootstrap_rng.randrange(len(image_ids)) for _ in image_ids)
        denominator = sum(cluster_sizes[index] * multiplicity for index, multiplicity in counts.items())
        for method in method_names:
            numerator = sum(cluster_deltas[method][index] * multiplicity for index, multiplicity in counts.items())
            draws[method].append(100.0 * numerator / denominator)
    for method in method_names:
        target = summary["actions"].get(method, summary["baselines"].get(method))
        target["delta_ci_95_pp"] = [_quantile(draws[method], 0.025), _quantile(draws[method], 0.975)]
    return summary


def write_report(summary: dict, path: Path) -> None:
    """Write a concise Markdown report of measured results, never placeholders."""
    path = Path(path)
    coverage = summary["coverage"]
    lines = [
        "# LookAgain paired evaluation", "",
        f"Status: **{summary['status']}**. Complete pairs: "
        f"**{coverage['complete_examples']}/{coverage['example_groups_seen']} examples** "
        f"across **{coverage['complete_images']} images**. "
        f"Excluded examples: **{coverage['excluded_examples']}**.", "",
        coverage["scope"], "",
    ]
    for warning in summary["warnings"]:
        lines.extend([f"**Note:** {warning}", ""])
    methods = [(action, metrics) for action, metrics in summary["actions"].items()]
    methods.extend((metrics["label"], metrics) for metrics in summary["baselines"].values())
    if methods:
        lines.extend([
            "Accuracy and changes use identical complete examples. Delta and its 95% CI are in percentage points relative to direct.", "",
            "| Method | Accuracy | Delta [95% CI] | Fixes / harms | Mean / median / p95 seconds | Max GiB |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for name, metrics in methods:
            interval = metrics["delta_ci_95_pp"]
            ci = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval is not None else "[unavailable]"
            latency = metrics["total_latency_s"]
            lines.append(
                f"| {name} | {100 * metrics['accuracy']:.2f}% | {metrics['delta_accuracy_pp']:+.2f} {ci} | "
                f"{metrics['wrong_to_right_count']} / {metrics['right_to_wrong_count']} | "
                f"{latency['mean']:.3f} / {latency['median']:.3f} / {latency['p95']:.3f} | "
                f"{metrics['policy_peak_memory_gib']['max']:.2f} |"
            )
        lines.extend([
            "", "Fixes are wrong-to-right transitions; harms are right-to-wrong transitions. "
            "The bootstrap resamples whole images, retaining all questions and paired actions; "
            f"{summary['bootstrap']['samples']} replicates, seed {summary['bootstrap']['seed']}. "
            "Intervals describe data sampling, not generation variability, and are not adjusted for multiple comparisons.", "",
            "Direct and highres are standalone runs. Every other action includes direct latency plus its incremental latency. "
            "Sequential peak memory is the maximum, not the sum. Selection/proposal overhead is included only if recorded.", "",
        ])
        if "uniform_random_crop" in summary["baselines"]:
            lines.extend(["Random crop uses one uniform, fixed seeded choice per example; it is not tuned on these outcomes. "
                          "Its interval is conditional on that draw.", ""])
        lines.extend(["The hindsight oracle uses answer correctness to select the cheapest correct available action, "
                      "or stops when all actions are wrong. It excludes highres and charges only the selected candidate. "
                      "It is an optimistic diagnostic, not a deployable controller.", ""])
    if coverage["excluded_examples"]:
        lines.extend([
            "## Coverage issues", "",
            "| Action | Missing examples | Duplicate examples | Failed records | Invalid records |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for action, counts in coverage["per_action"].items():
            lines.append(f"| {action} | {counts['missing_examples']} | {counts['duplicate_examples']} | "
                         f"{counts['failed_records']} | {counts['invalid_records']} |")
        lines.extend(["", "Full excluded-example reasons are available in the analysis summary's coverage.excluded_groups.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
