"""Prespecified secondary diagnostics; no model loading or answer rescoring.

Run from any directory with --run RUN_DIRECTORY --output OUTPUT_DIRECTORY.
All oracle selections are privileged hindsight diagnostics, never policies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lookagain.actions import normalize_answer
from lookagain.analysis import analyze_records

BOOTSTRAP_SAMPLES = 10000
SEED = 20260925
SENTENCE_RULE = (
    "Parsed answer has at least 6 whitespace-separated words, begins with yes/no "
    "followed by comma/semicolon/colon, or begins with it/there/this/that/they/these/those "
    "followed by is/are/was/were/has/have. A descriptive heuristic, not a grammar or correctness judge."
)
SENTENCE_PREFIX = re.compile(
    r"^(?:(?:yes|no)[,;:]|(?:it|there|this|that|they|these|those)\s+(?:is|are|was|were|has|have)\b)", re.I
)
CAVEATS = [
    "All oracles use recorded ground-truth correctness to choose among actions. They are privileged "
    "opportunity bounds, not feasible controllers; searching more actions itself increases this bound.",
    "Exact-match grading and annotation ambiguity can mark semantically acceptable answers wrong. "
    "The original grades are preserved; format and sentence indicators never rescore an answer.",
    "An exclusive crop rescue is an outcome in these recorded branches, not causal proof that new "
    "visual evidence helped. Prompts, repeated attempts, representation and compute may also matter.",
    "The nonvisual oracle means no new image region: its recheck action still repeats the overview image. "
    "Standalone highres is compared separately and is excluded from every follow-up oracle.",
    "This is a development convenience sample, not a held-out or representative benchmark. "
    "Image-bootstrap intervals describe conditional sampling uncertainty, not generation variability, "
    "and are not adjusted for multiple comparisons. No confidence threshold is tuned here.",
    "Upsampling introduces no new source detail. This report makes no efficiency claim; use the main "
    "paired report for measured cumulative latency and memory. Image uniqueness checks IDs only.",
]


def _percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _comparisons(vectors, image_ids):
    reference = vectors["oracle_nonvisual"]
    names = ("oracle_stop_plus_crops", "oracle_all")
    differences = {name: [int(a) - int(b) for a, b in zip(vectors[name], reference)] for name in names}
    result = {name: {
        "reference": "oracle_nonvisual", "delta_accuracy_pp": 100 * sum(values) / len(values),
        "wins": values.count(1), "losses": values.count(-1), "delta_ci_95_pp": None,
    } for name, values in differences.items()}
    indices = defaultdict(list)
    for position, image_id in enumerate(image_ids):
        indices[image_id].append(position)
    if len(indices) < 2:
        return result
    clusters = [(len(positions), *(sum(differences[name][i] for i in positions) for name in names))
                for _, positions in sorted(indices.items())]
    rng = random.Random(SEED)
    draws = {name: [] for name in names}
    for _ in range(BOOTSTRAP_SAMPLES):
        counts = Counter(rng.randrange(len(clusters)) for _ in clusters)
        denominator = sum(clusters[i][0] * count for i, count in counts.items())
        for column, name in enumerate(names, start=1):
            draws[name].append(100 * sum(clusters[i][column] * count for i, count in counts.items()) / denominator)
    for name in names:
        result[name]["delta_ci_95_pp"] = [_percentile(draws[name], p) for p in (0.025, 0.975)]
    return result


def diagnose(run: Path) -> dict:
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    actions = metadata["config"]["actions"]
    crops = sorted(action for action in actions if action.startswith("crop_"))
    if not {"direct", "highres", "think", "recheck"}.issubset(actions) or not crops:
        raise ValueError("This diagnostic requires direct, highres, think, recheck and at least one crop action.")
    planned_list = metadata["example_ids"]
    if not planned_list or any(not isinstance(item, str) or not item for item in planned_list) or len(set(planned_list)) != len(planned_list):
        raise ValueError("run.json must declare a nonempty list of unique example IDs.")
    # Reuse authoritative complete-group validation, without running its full bootstrap.
    paired = analyze_records(records, bootstrap_samples=1, expected_actions=actions)
    planned = set(planned_list)
    seen = {row["example_id"] for row in records}
    if seen - planned:
        raise ValueError("Records contain examples outside the declared run.")
    coverage = paired["coverage"]
    excluded = {row["example_id"] for row in coverage["excluded_groups"]}
    grouped = defaultdict(dict)
    for row in records:
        if row["example_id"] not in excluded and row["action"] in actions:
            grouped[row["example_id"]][row["action"]] = row
    examples = sorted(grouped)
    groups = [grouped[example] for example in examples]
    image_ids = [group["direct"]["image_id"] for group in groups]
    duplicate_images = {key: count for key, count in Counter(image_ids).items() if count > 1}
    coverage.update(planned_examples=len(planned), wholly_absent_examples=sorted(planned - seen),
                    planned_completion_fraction=len(groups) / len(planned),
                    one_example_per_complete_image=not duplicate_images,
                    duplicate_complete_image_ids=duplicate_images)
    warnings = list(paired["warnings"])
    if planned - seen:
        warnings.append("Some planned examples have no records; all diagnostics describe complete groups only.")
    if duplicate_images:
        warnings.append("Repeated image IDs detected; bootstrap still resamples entire image clusters.")
    methods = {
        "direct": ["direct"], "highres": ["highres"],
        "oracle_stop_plus_crops": ["direct", *crops],
        "oracle_nonvisual": ["direct", "think", "recheck"],
        "oracle_all": ["direct", "think", "recheck", *crops],
    }
    result = {
        "diagnostic_spec_version": 1, "run_fingerprint": metadata["fingerprint"],
        "diagnostic_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "complete" if len(groups) == len(planned) else "incomplete_coverage",
        "coverage": coverage, "warnings": warnings,
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": SEED, "unit": "image_id",
                      "interval": "95% percentile", "estimator": "example-weighted paired accuracy difference"},
        "methods": {}, "comparisons_vs_nonvisual_oracle": {}, "crop_exclusive_rescues": {},
        "format_diagnostics": {}, "crop_answer_changes": {},
        "sentence_indicator_definition": SENTENCE_RULE, "caveats": CAVEATS,
    }
    if not groups:
        return result
    vectors = {name: [any(group[action]["correct"] for action in candidates) for group in groups]
               for name, candidates in methods.items()}
    result["methods"] = {name: {"candidate_actions": methods[name], "privileged": name.startswith("oracle_"),
                                "correct_count": sum(values), "n_examples": len(groups),
                                "accuracy": sum(values) / len(groups)} for name, values in vectors.items()}
    result["comparisons_vs_nonvisual_oracle"] = _comparisons(vectors, image_ids)
    no_region_wrong = [i for i, correct in enumerate(vectors["oracle_nonvisual"]) if not correct]
    exclusive = [i for i in no_region_wrong if vectors["oracle_stop_plus_crops"][i]]
    cross = Counter((vectors["oracle_stop_plus_crops"][i], vectors["highres"][i]) for i in no_region_wrong)
    result["crop_exclusive_rescues"] = {
        "definition": "At least one crop is correct while direct, think and recheck are all incorrect.",
        "count": len(exclusive), "n_examples": len(groups), "fraction_all_examples": len(exclusive) / len(groups),
        "nonvisual_all_wrong_count": len(no_region_wrong),
        "fraction_of_nonvisual_all_wrong": len(exclusive) / len(no_region_wrong) if no_region_wrong else None,
        "example_ids": [examples[i] for i in exclusive],
        "highres_correct_among_exclusive": sum(vectors["highres"][i] for i in exclusive),
        "highres_wrong_among_exclusive": sum(not vectors["highres"][i] for i in exclusive),
        "contingency_among_nonvisual_all_wrong": {
            "crop_rescue_highres_correct": cross[(True, True)],
            "crop_rescue_highres_wrong": cross[(True, False)],
            "no_crop_rescue_highres_correct": cross[(False, True)],
            "no_crop_rescue_highres_wrong": cross[(False, False)],
        },
    }
    for action in actions:
        rows = [group[action] for group in groups]
        parsed = [row for row in rows if row.get("parse_valid") is True and isinstance(row.get("predicted_answer"), str)]
        sentence_rows = [row for row in parsed if len(row["predicted_answer"].split()) >= 6
                         or SENTENCE_PREFIX.search(row["predicted_answer"].strip())]
        result["format_diagnostics"][action] = {
            "n_examples": len(rows), "invalid_answers": sum(row.get("parse_valid") is False for row in rows),
            "parse_flag_missing_or_invalid": sum(not isinstance(row.get("parse_valid"), bool) for row in rows),
            "truncated_generations": sum(row.get("generation_truncated") is True for row in rows),
            "truncation_flag_missing_or_invalid": sum(not isinstance(row.get("generation_truncated"), bool) for row in rows),
            "answer_format_counts": dict(sorted(Counter(str(row.get("answer_format", "unknown")) for row in rows).items())),
            "parsed_answer_count": len(parsed), "sentence_like_answers": len(sentence_rows),
            "sentence_like_exact_match_wrong": sum(not row["correct"] for row in sentence_rows),
        }
    for action in crops:
        comparable = [(group["direct"], group[action]) for group in groups
                      if all(row.get("parse_valid") is True and isinstance(row.get("predicted_answer"), str)
                             for row in (group["direct"], group[action]))]
        changed = [(base, crop) for base, crop in comparable
                   if normalize_answer(base["predicted_answer"]) != normalize_answer(crop["predicted_answer"])]
        result["crop_answer_changes"][action] = {
            "comparable_parsed_pairs": len(comparable), "uncomparable_pairs": len(groups) - len(comparable),
            "normalized_answer_changes": len(changed), "normalized_answer_unchanged": len(comparable) - len(changed),
            "changed_wrong_to_right": sum(not base["correct"] and crop["correct"] for base, crop in changed),
            "changed_right_to_wrong": sum(base["correct"] and not crop["correct"] for base, crop in changed),
            "changed_both_wrong": sum(not base["correct"] and not crop["correct"] for base, crop in changed),
            "changed_both_right": sum(base["correct"] and crop["correct"] for base, crop in changed),
        }
    return result


def markdown(result: dict) -> str:
    coverage = result["coverage"]
    lines = ["# LookAgain secondary diagnostic", "",
             f"Status: **{result['status']}**. Complete groups: **{coverage['complete_examples']}/{coverage['planned_examples']} planned examples**. "
             f"Wholly absent: {len(coverage['wholly_absent_examples'])}; excluded observed groups: {coverage['excluded_examples']}.", ""]
    for warning in result["warnings"]:
        lines.extend([f"**Note:** {warning}", ""])
    if result["methods"]:
        lines.extend(["Every oracle below is a privileged hindsight bound, not a feasible policy.", "",
                      "| Method | Correct / examples | Accuracy |", "| --- | ---: | ---: |"])
        for name, metric in result["methods"].items():
            lines.append(f"| {name} | {metric['correct_count']} / {metric['n_examples']} | {100 * metric['accuracy']:.2f}% |")
        lines.extend(["", "| Compared with oracle_nonvisual | Difference, pp [95% image-bootstrap CI] | Wins / losses |",
                      "| --- | ---: | ---: |"])
        for name, metric in result["comparisons_vs_nonvisual_oracle"].items():
            interval = metric["delta_ci_95_pp"]
            ci = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "[unavailable]"
            lines.append(f"| {name} | {metric['delta_accuracy_pp']:+.2f} {ci} | {metric['wins']} / {metric['losses']} |")
        rescue = result["crop_exclusive_rescues"]
        lines.extend(["", f"Crop-exclusive rescues: **{rescue['count']}** out of {rescue['n_examples']} examples; "
                      f"{rescue['nonvisual_all_wrong_count']} examples had direct, think and recheck all wrong. "
                      f"Among exclusive rescues, standalone highres was correct in {rescue['highres_correct_among_exclusive']} "
                      f"and wrong in {rescue['highres_wrong_among_exclusive']}.", "",
                      "| Action | Invalid format | Truncated | Sentence-like parsed answers | Of these, exact-match wrong |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for name, row in result["format_diagnostics"].items():
            lines.append(f"| {name} | {row['invalid_answers']} | {row['truncated_generations']} | {row['sentence_like_answers']} | {row['sentence_like_exact_match_wrong']} |")
        lines.extend(["", "Sentence-like definition: " + SENTENCE_RULE, "",
                      "| Crop | Comparable parsed pairs | Changed | Unchanged | Changed fixes / harms / both wrong |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for name, row in result["crop_answer_changes"].items():
            lines.append(f"| {name} | {row['comparable_parsed_pairs']} | {row['normalized_answer_changes']} | "
                         f"{row['normalized_answer_unchanged']} | {row['changed_wrong_to_right']} / {row['changed_right_to_wrong']} / {row['changed_both_wrong']} |")
        lines.extend(["", f"Bootstrap: {BOOTSTRAP_SAMPLES} image-cluster resamples, seed {SEED}. "
                      "All diagnostics use the same complete paired cohort. Detailed format counts, missing flags, "
                      "uncomparable pairs, exclusive example IDs and the highres contingency table are in diagnostics.json.", ""])
    lines.extend(f"- {caveat}" for caveat in result["caveats"])
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.run.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output / "diagnostics.md").write_text(markdown(result), encoding="utf-8")
    print(args.output / "diagnostics.md")


if __name__ == "__main__":
    main()
