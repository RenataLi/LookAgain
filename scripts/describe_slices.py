"""Descriptive slices and transitions for an entirely completed declared run.

Requires only the Python standard library, Pillow, and this repository's
lookagain.analysis module. No models, answer rescoring, inferential statistics,
or slice selection are used. A run may declare a subset of a larger manifest;
the SHA-256 of that full manifest must still match its recorded run identity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lookagain.analysis import DEFAULT_ACTIONS, analyze_records


IDENTITY_KEYS = ("config", "manifest_sha256", "code_sha256", "model", "runtime", "example_ids")
SIZE_BINS = ("<=640", "641..1024", ">1024")
CAVEATS = [
    "This is a descriptive report of one completed declared run on a development convenience sample, "
    "not a representative or held-out benchmark evaluation.",
    "Every declared question type and all three fixed native-image-size bins are reported. "
    "Rows are not selected or ranked by observed performance; small cells can be unstable.",
    "No confidence intervals, p-values, hypothesis tests, or multiple-comparison claims are provided. "
    "The single bootstrap draw used inside the shared validator is discarded and is not an uncertainty estimate.",
    "Fixes and harms refer to the existing recorded correctness labels relative to direct. "
    "Answers are not rescored; exact-match or annotation limitations remain.",
    "Conditional fix rate is fixes divided by direct-incorrect examples; conditional harm rate is harms "
    "divided by direct-correct examples. A zero denominator produces null, not zero.",
    "Size bins use the native saved image long edge, not inference-time resized dimensions. "
    "A larger canvas does not create source pixels. Differences do not establish causal effects, "
    "new visual evidence, efficiency, or out-of-distribution generalization.",
    "Fingerprint verification checks the recorded run identity. Source-file hashes identify code; "
    "they do not retroactively prove which program generated every record.",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path.name}, line {number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected object in {path.name}, line {number}")
        rows.append(row)
    return rows


def _nonempty_unique(values: list, label: str) -> None:
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def transitions(groups: list[dict], actions: list[str]) -> dict:
    """Retain all four paired outcomes, explicit denominators, and null rates."""
    count = len(groups)
    direct_correct = sum(group["direct"]["correct"] for group in groups)
    direct_incorrect = count - direct_correct
    result = {}
    for action in actions:
        cross = Counter((group["direct"]["correct"], group[action]["correct"]) for group in groups)
        fixes, harms = cross[(False, True)], cross[(True, False)]
        correct_count = cross[(True, True)] + fixes
        result[action] = {
            "n_examples": count,
            "correct_count": correct_count,
            "accuracy": _fraction(correct_count, count),
            "direct_correct_count": direct_correct,
            "direct_incorrect_count": direct_incorrect,
            "correct_to_correct": cross[(True, True)],
            "wrong_to_right": fixes,
            "right_to_wrong": harms,
            "wrong_to_wrong": cross[(False, False)],
            "conditional_fix_rate": _fraction(fixes, direct_incorrect),
            "conditional_fix_rate_numerator": fixes,
            "conditional_fix_rate_denominator": direct_incorrect,
            "conditional_harm_rate": _fraction(harms, direct_correct),
            "conditional_harm_rate_numerator": harms,
            "conditional_harm_rate_denominator": direct_correct,
        }
    return result


def describe(run: Path, manifest: Path) -> dict:
    run, manifest = Path(run).resolve(), Path(manifest).resolve()
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or any(key not in metadata for key in IDENTITY_KEYS):
        raise ValueError("run.json is missing required run-identity fields")
    identity = {key: metadata[key] for key in IDENTITY_KEYS}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("run.json fingerprint does not match its recorded identity")
    manifest_hash = sha256_file(manifest)
    if manifest_hash != metadata["manifest_sha256"]:
        raise ValueError("Manifest SHA-256 does not match run.json")
    planned = metadata["example_ids"]
    if not isinstance(planned, list):
        raise ValueError("run.json example_ids must be a list")
    _nonempty_unique(planned, "Planned example IDs")
    actions = metadata["config"].get("actions")
    if not isinstance(actions, list) or len(actions) != len(DEFAULT_ACTIONS) or set(actions) != set(DEFAULT_ACTIONS):
        raise ValueError("This report requires all eight declared LookAgain actions exactly once")
    # Fixed presentation order, independent of accuracy or input record order.
    actions = ["direct", "highres", "think", "recheck", "crop_tl", "crop_tr", "crop_bl", "crop_br"]
    crops = [action for action in actions if action.startswith("crop_")]

    manifest_rows = read_jsonl(manifest)
    _nonempty_unique([row.get("example_id") for row in manifest_rows], "Manifest example IDs")
    _nonempty_unique([row.get("image_id") for row in manifest_rows], "Manifest image IDs")
    by_example = {row["example_id"]: row for row in manifest_rows}
    if set(planned) - set(by_example):
        raise ValueError("Some planned examples do not exist in the matching manifest")

    records_path = run / "records.jsonl"
    records = read_jsonl(records_path)
    seen = {row.get("example_id") for row in records}
    if seen != set(planned):
        raise ValueError("Run is incomplete or contains undeclared examples; report requires the full planned cohort")
    # Shared validator checks duplicate/missing actions, status, booleans and finite costs.
    paired = analyze_records(records, bootstrap_samples=1, expected_actions=actions)
    coverage = paired["coverage"]
    if (paired["status"] != "complete" or coverage["complete_examples"] != len(planned)
            or coverage["excluded_examples"] or coverage["ignored_action_records"]
            or len(records) != len(planned) * len(actions)):
        raise ValueError("Run must have one valid successful record for every planned example/action")
    grouped = defaultdict(dict)
    for row in records:
        source = by_example[row["example_id"]]
        if row["image_id"] != source["image_id"]:
            raise ValueError(f"Record/manifest image mismatch: {row['example_id']}")
        if row.get("question") != source.get("question") or row.get("target_answer") != source.get("answer"):
            raise ValueError(f"Record/manifest question or target mismatch: {row['example_id']}")
        grouped[row["example_id"]][row["action"]] = row
    _nonempty_unique([grouped[key]["direct"]["image_id"] for key in planned], "Completed image IDs")

    dimensions = {}
    pixel_hashes: dict[str, str] = {}
    encoded_hashes: dict[str, str] = {}
    slice_groups = {"question_type": defaultdict(list), "native_long_edge": {key: [] for key in SIZE_BINS}}
    for example_id in planned:
        source = by_example[example_id]
        relative = source.get("image_path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or PureWindowsPath(relative).drive:
            raise ValueError(f"Image path must be relative: {example_id}")
        image_path = (manifest.parent / relative).resolve()
        if not image_path.is_relative_to(manifest.parent):
            raise ValueError(f"Image path escapes manifest directory: {example_id}")
        image_hash = sha256_file(image_path)
        if image_hash != source.get("image_sha256"):
            raise ValueError(f"Image SHA-256 mismatch: {example_id}")
        if image_hash in encoded_hashes:
            raise ValueError(f"Duplicate encoded source images: {encoded_hashes[image_hash]}, {example_id}")
        encoded_hashes[image_hash] = example_id
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            pixel_hash = hashlib.sha256(f"RGB:{width}:{height}:".encode() + image.convert("RGB").tobytes()).hexdigest()
        if pixel_hash in pixel_hashes:
            raise ValueError(f"Duplicate decoded source images: {pixel_hashes[pixel_hash]}, {example_id}")
        pixel_hashes[pixel_hash] = example_id
        for action, record in grouped[example_id].items():
            if "source_size" in record and record["source_size"] != [width, height]:
                raise ValueError(f"Recorded source dimensions disagree: {example_id}/{action}")
        dimensions[example_id] = [width, height]
        long_edge = max(width, height)
        size_bin = "<=640" if long_edge <= 640 else "641..1024" if long_edge <= 1024 else ">1024"
        question_type = source.get("question_type") or "missing"
        if not isinstance(question_type, str):
            raise ValueError(f"question_type must be a string: {example_id}")
        slice_groups["question_type"][question_type].append(grouped[example_id])
        slice_groups["native_long_edge"][size_bin].append(grouped[example_id])

    groups = [grouped[key] for key in planned]
    overall = transitions(groups, actions)
    # Point counts agree with the project's authoritative paired analysis.
    for action in actions:
        metric = paired["actions"][action]
        if (overall[action]["wrong_to_right"] != metric["wrong_to_right_count"]
                or overall[action]["right_to_wrong"] != metric["right_to_wrong_count"]
                or overall[action]["accuracy"] != metric["accuracy"]):
            raise RuntimeError("Descriptive transitions disagree with shared analysis")
    slices = {}
    for dimension, bins in slice_groups.items():
        labels = sorted(bins) if dimension == "question_type" else SIZE_BINS
        slices[dimension] = [
            {"slice": label, "n_examples": len(bins[label]), "actions": transitions(bins[label], actions)}
            for label in labels
        ]
    return {
        "schema_version": 1,
        "status": "complete",
        "purpose": "descriptive_only",
        "run_fingerprint": fingerprint,
        "manifest_sha256": manifest_hash,
        "records_sha256": sha256_file(records_path),
        "report_script_sha256": sha256_file(Path(__file__)),
        "shared_analysis_sha256": sha256_file(Path(__file__).resolve().parents[1] / "src/lookagain/analysis.py"),
        "source_code_sha256_from_run": metadata["code_sha256"],
        "provenance": {
            "datasets": sorted({row.get("dataset", "missing") for row in manifest_rows}),
            "dataset_revisions": sorted({row.get("dataset_revision", "missing") for row in manifest_rows}),
            "source_splits": sorted({row.get("source_split", "missing") for row in manifest_rows}),
            "manifest_examples": len(manifest_rows),
            "planned_examples": len(planned),
            "complete_examples": len(groups),
            "unique_complete_images": len(groups),
            "source_image_hashes_verified": len(groups),
            "no_exact_encoded_or_decoded_duplicates_in_planned_cohort": True,
            "manifest_subset_allowed": True,
        },
        "actions": actions,
        "crop_actions": crops,
        "overall_transitions": overall,
        "slices": slices,
        "slice_definitions": {
            "question_type": "manifest question_type; missing values form a missing row; alphabetical order",
            "native_long_edge": "max(original saved image width,height) in pixels; <=640,641..1024,>1024",
        },
        "inferential_statistics_reported": False,
        "caveats": CAVEATS,
    }


def _pct(value: float | None) -> str:
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def _rate_cell(row: dict, prefix: str) -> str:
    return f"{row[prefix + '_numerator']} / {row[prefix + '_denominator']} ({_pct(row[prefix])})"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def markdown(result: dict) -> str:
    provenance = result["provenance"]
    actions, crops = result["actions"], result["crop_actions"]
    lines = [
        "# LookAgain descriptive slices and transitions", "",
        f"Complete declared run: **{provenance['complete_examples']}/{provenance['planned_examples']} examples**, "
        f"each from a unique source image. The matching manifest contains {provenance['manifest_examples']} examples; "
        "only the run's declared cohort is described.", "",
        "Fingerprint, manifest hash, selected-image hashes, unique source-image IDs and complete action coverage "
        "were verified. All correctness labels are the original recorded labels.", "",
        f"- Run fingerprint: `{result['run_fingerprint']}`.",
        f"- Manifest SHA-256: `{result['manifest_sha256']}`.",
        f"- Records SHA-256: `{result['records_sha256']}`.", "",
        "## Overall paired transitions", "",
        "Fix rate conditions on direct being wrong; harm rate conditions on direct being correct. "
        "Explicit denominators are shown. A zero denominator is unavailable (JSON null). "
        "The direct row is the identity comparison; highres is a separate standalone action.", "",
        "| Action | N | Accuracy | Correct→correct | Wrong→wrong | Fixes / direct wrong (rate) | Harms / direct correct (rate) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for action in actions:
        row = result["overall_transitions"][action]
        lines.append(f"| {action} | {row['n_examples']} | {_pct(row['accuracy'])} | {row['correct_to_correct']} | "
                     f"{row['wrong_to_wrong']} | {_rate_cell(row, 'conditional_fix_rate')} | "
                     f"{_rate_cell(row, 'conditional_harm_rate')} |")
    for dimension, title in [("question_type", "Detailed question type"), ("native_long_edge", "Native image long edge, pixels")]:
        lines.extend(["", f"## {title}", "", "Accuracy uses the identical examples within each row. "
                      "All declared categories are retained; zero-sized fixed bins are shown as unavailable.", "",
                      "| Slice | N | " + " | ".join(actions) + " |",
                      "| --- | ---: | " + " | ".join("---:" for _ in actions) + " |"])
        for row in result["slices"][dimension]:
            lines.append(f"| {_escape(row['slice'])} | {row['n_examples']} | "
                         + " | ".join(_pct(row["actions"][action]["accuracy"]) for action in actions) + " |")
        lines.extend(["", "Crop transitions below are counts relative to direct on the same slice; each cell is fixes / harms.", "",
                      "| Slice | N | " + " | ".join(crops) + " |",
                      "| --- | ---: | " + " | ".join("---:" for _ in crops) + " |"])
        for row in result["slices"][dimension]:
            lines.append(f"| {_escape(row['slice'])} | {row['n_examples']} | "
                         + " | ".join(f"{row['actions'][action]['wrong_to_right']} / {row['actions'][action]['right_to_wrong']}"
                                      for action in crops) + " |")
    lines.extend(["", "## Scope and limitations", ""])
    lines.extend(f"- {caveat}" for caveat in result["caveats"])
    lines.extend(["", "`slices.json` contains all transition counts, explicit denominators, conditional rates, "
                  "slice definitions and file hashes. No best-performing slice is selected.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for slices.json and slices.md")
    args = parser.parse_args()
    # Do all validation before creating or writing the report output.
    result = describe(args.run, args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "slices.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "slices.md").write_text(markdown(result), encoding="utf-8")
    print(f"Descriptive report: {args.output / 'slices.md'}")
    print(f"Validated all {result['provenance']['complete_examples']} planned examples; no inferential statistics reported.")


if __name__ == "__main__":
    main()
