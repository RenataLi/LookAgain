"""Compare a separate warm repeat with its matched examples in the original run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lookagain.analysis import analyze_records


def read_run(path):
    metadata = json.loads((path / "run.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (path / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    checked = analyze_records(records, bootstrap_samples=1, expected_actions=metadata["config"]["actions"])
    planned = set(metadata["example_ids"])
    if checked["status"] != "complete" or checked["coverage"]["complete_examples"] != len(planned):
        raise ValueError("Both runs must be complete")
    if {r["example_id"] for r in records} != planned:
        raise ValueError("Recorded and planned IDs disagree")
    return metadata, {(r["example_id"], r["action"]): r for r in records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original, first = read_run(args.reference)
    repeated, second = read_run(args.repeat)
    for key in ("config", "manifest_sha256", "code_sha256", "model", "runtime"):
        if original[key] != repeated[key]:
            raise ValueError(f"Incompatible run field: {key}")
    ids = repeated["example_ids"]
    if ids != original["example_ids"][:len(ids)]:
        raise ValueError("The repeat must use the declared first-N manifest subset")
    rows = {}
    for action in original["config"]["actions"]:
        def costs(records):
            return [records[(i, action)]["elapsed_s"] +
                    (records[(i, "direct")]["elapsed_s"] if action not in ("direct", "highres") else 0)
                    for i in ids]
        before, after = costs(first), costs(second)
        rows[action] = {
            "n_examples": len(ids),
            "original_total_mean_s": statistics.mean(before),
            "repeat_total_mean_s": statistics.mean(after),
            "original_total_median_s": statistics.median(before),
            "repeat_total_median_s": statistics.median(after),
            "ratio_of_means_repeat_over_original": statistics.mean(after) / statistics.mean(before),
            "changed_raw_responses": sum(first[(i, action)]["response"] != second[(i, action)]["response"] for i in ids),
            "changed_exact_match_grades": sum(first[(i, action)]["correct"] != second[(i, action)]["correct"] for i in ids),
        }
    result = {
        "reference_fingerprint": original["fingerprint"], "repeat_fingerprint": repeated["fingerprint"],
        "example_ids": ids, "actions": rows,
        "scope": "One separate first-N repeat after the main run. Descriptive timing check only; not pooled into primary accuracy.",
        "cost": "Direct/highres: single call. All follow-ups: own call plus initial direct call. Setup and warmup excluded.",
        "limitations": "Small prefix, one repetition, same deterministic action order per example, uncontrolled background host activity. No general speedup or variance claim.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "timing_repeat.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Separate timing repeat", "", result["scope"], "", result["cost"], "",
             f"Matched examples: {len(ids)}. Both runs have identical model/configuration/source/runtime and complete declared action coverage.", "",
             "| Action | Original mean s | Repeat mean s | Repeat/original | Changed responses / grades |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for action, row in rows.items():
        lines.append(f"| {action} | {row['original_total_mean_s']:.3f} | {row['repeat_total_mean_s']:.3f} | {row['ratio_of_means_repeat_over_original']:.2f} | {row['changed_raw_responses']} / {row['changed_exact_match_grades']} |")
    lines.extend(["", result["limitations"], "", "Raw repeat records and metadata accompany this table; all main results use only the original full panel.", ""])
    (args.output / "timing_repeat.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output / "timing_repeat.md")


if __name__ == "__main__":
    main()
