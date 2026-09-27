"""Prepare, freeze and unblind a descriptive DUDE answer-pair source review.

The public local packet contains every changed primary pair, independently of
automatic scores. The condition key must remain outside the packet directory.
This workflow never changes automatic grades or creates reviewer judgments.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import random

from PIL import Image

SEED = 20260927
PAIR_ACTIONS = ("native_256", "degraded_256")
CATEGORIES = {
    "content_difference", "formatting_only", "both_substantively_valid",
    "both_substantively_wrong", "ambiguous_uncertain",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    require(isinstance(value, str), "A timezone-aware timestamp is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Invalid timestamp") from error
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "Naive timestamps are not accepted")
    return parsed


def relative_file(root, name):
    require(isinstance(name, str), "Relative path must be a string")
    rel, win = Path(name), PureWindowsPath(name)
    target = (Path(root) / rel).resolve()
    require(not rel.is_absolute() and not win.is_absolute() and not win.drive
            and ".." not in win.parts and target.is_relative_to(Path(root).resolve()), "Unsafe relative asset path")
    return target


def changed_pair(first, second):
    """Exact parsed changes; plus raw changes whenever either parse is invalid."""
    for item in (first, second):
        require(isinstance(item["response"], str) and type(item["parse_valid"]) is bool, "Invalid response schema")
        require((item["parse_valid"] and isinstance(item["predicted_answer"], str))
                or (not item["parse_valid"] and item["predicted_answer"] is None), "Inconsistent parsed response")
    return (first["predicted_answer"] != second["predicted_answer"]
            or (not (first["parse_valid"] and second["parse_valid"])
                and first["response"] != second["response"]))


def assignment(example_id):
    """Stable per-source swap, independent of scores, answers and iteration order."""
    digest = hashlib.sha256(f"dude-semantic-blind:{SEED}:{example_id}".encode("utf-8")).digest()
    actions = list(PAIR_ACTIONS)
    random.Random(int.from_bytes(digest, "big")).shuffle(actions)
    return tuple(actions)


def case_order(example_id):
    return hashlib.sha256(f"dude-semantic-case-order:{SEED}:{example_id}".encode("utf-8")).hexdigest()


def _validated_run(manifest, lock_path, run_dir):
    # Lazy import: packet-only helpers/tests do not load a runner or model.
    from dude_replication import validate_completed_run
    return validate_completed_run(Path(manifest), Path(lock_path), Path(run_dir), role="main")


def prepare(manifest: Path, lock_path: Path, run_dir: Path, output: Path, key_path: Path):
    manifest, lock_path, run_dir, output, key_path = map(lambda p: Path(p).resolve(),
                                                       (manifest, lock_path, run_dir, output, key_path))
    require(not output.exists() and not key_path.exists(), "Use new packet and private-key paths")
    require(not key_path.is_relative_to(output) and not output.is_relative_to(key_path), "Private condition key must be outside packet directory")
    require(not output.is_relative_to(Path(__file__).resolve().parents[1]), "Keep local source-image packets outside the public code project")
    rows, records, run, completed = _validated_run(manifest, lock_path, run_dir)
    input_paths = {"manifest_sha256": manifest, "lock_sha256": lock_path,
                   "run_sha256": run_dir / "run.json", "records_sha256": run_dir / "records.jsonl",
                   "completed_sha256": run_dir / "completed.json"}
    input_hashes = {name: sha(path) for name, path in input_paths.items()}
    require(len({row["example_id"] for row in rows}) == len(rows), "Duplicate validated source IDs")
    require(len({row["source_cluster_id"] for row in rows}) == len(rows), "Duplicate validated source clusters")
    selected = []
    for row in rows:
        pair = [records[row["example_id"], action] for action in PAIR_ACTIONS]
        if changed_pair(*pair):
            selected.append(row)
    selected.sort(key=lambda row: (case_order(row["example_id"]), row["example_id"]))
    # Validate all source bindings and ROI bounds before creating a partial pack.
    for row in selected:
        source_path = relative_file(manifest.parent, row["image_path"])
        require(sha(source_path) == row["image_sha256"], "Source image hash differs")
        with Image.open(source_path) as image:
            box = row["roi_pixels"]
            require(len(box) == 4 and all(type(value) is int for value in box), "ROI must have four integer coordinates")
            require(0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height, "ROI outside source image")
        require(len(row["original_answers"]) == 1 and row["original_answers"][0] in row["validated_primary_answers"],
                "Source-validated canonical reference is required")
    output.mkdir(parents=True)
    cases, keys = [], []
    for index, row in enumerate(selected, 1):
        case_id = f"case{index:03d}"
        first, second = assignment(row["example_id"])
        a, b = records[row["example_id"], first], records[row["example_id"], second]
        page, roi = f"{case_id}-page.png", f"{case_id}-region.png"
        with Image.open(relative_file(manifest.parent, row["image_path"])) as image:
            source = image.convert("RGB")
            source.save(output / page)
            source.crop(row["roi_pixels"]).save(output / roi)
        case = {"case_id": case_id, "question": row["question"],
                "reference_original": row["original_answers"][0],
                "reference_variants_original": row["original_answer_variants"],
                "references_source_validated": row["validated_primary_answers"],
                "full_page": {"path": page, "sha256": sha(output / page)},
                "evidence_region": {"path": roi, "sha256": sha(output / roi)},
                "raw_response_A": a["response"], "raw_response_B": b["response"],
                "parsed_answer_A": a["predicted_answer"], "parsed_answer_B": b["predicted_answer"]}
        case["pair_sha256"] = canonical_hash(case)
        cases.append(case)
        keys.append({"case_id": case_id, "example_id": row["example_id"],
                     "source_cluster_id": row["source_cluster_id"], "A": first, "B": second,
                     "pair_sha256": case["pair_sha256"], "source_image_sha256": row["image_sha256"]})
    private_payload = {"schema_version": 1, "seed": SEED, "role": "main", "keys": keys,
                       "assignment_method": "Python Random(SHA256(dude-semantic-blind:20260927:example_id) as big-endian integer).shuffle([native_256,degraded_256]); independent of responses/scores.",
                       "case_order_method": "Ascending SHA256(dude-semantic-case-order:20260927:example_id), then example_id; no manifest-order identifiers in public cases.",
                       "bindings": input_hashes}
    packet = {"schema_version": 1, "created_utc": now(), "condition_labels_hidden": True,
              "selection": "All exact parsed-answer changes in the paired comparison; additionally differing full raw responses when either parse is invalid. Equal-score pairs are included.",
              "assignment_commitment_sha256": canonical_hash(private_payload),
              "instructions": "Inspect the full page and evidence region with the original question and references. Judge both full raw answers as substantively correct, wrong or uncertain. Distinguish content changes from formatting, both valid, both wrong or ambiguity. Original variants are not automatically valid; source-approved references are identified separately. Do not open the condition key or automatic results before freezing the complete review. Review judgments never change automatic grades.",
              "categories": sorted(CATEGORIES), "cases": cases}
    packet_path = output / "packet.json"
    require(all(sha(path) == input_hashes[name] for name, path in input_paths.items()), "Validated inputs changed during packet preparation")
    write_new(packet_path, packet)
    write_new(key_path, {**private_payload, "packet_sha256": sha(packet_path)})
    return {"cases": len(cases), "packet": str(packet_path), "packet_sha256": sha(packet_path),
            "private_key": str(key_path), "new_model_calls": 0}


def validate_packet(packet_path):
    packet_path = Path(packet_path).resolve()
    packet = read(packet_path)
    require(packet.get("schema_version") == 1 and packet.get("condition_labels_hidden") is True, "Not a blinded packet")
    timestamp(packet["created_utc"])
    cases = packet["cases"]
    require([case["case_id"] for case in cases] == [f"case{i:03d}" for i in range(1, len(cases) + 1)], "Unexpected case identity/order")
    for case in cases:
        require(case["pair_sha256"] == canonical_hash({key: value for key, value in case.items() if key != "pair_sha256"}),
                "Packet pair content changed")
        for role, suffix in (("full_page", "page"), ("evidence_region", "region")):
            asset = case[role]
            require(asset["path"] == f"{case['case_id']}-{suffix}.png", "Nonblind image filename")
            require(sha(relative_file(packet_path.parent, asset["path"])) == asset["sha256"], "Packet image changed")
    return packet


def validate_decision(decision, case, packet_sha256, earliest, latest):
    require(decision.get("case_id") == case["case_id"] and decision.get("packet_sha256") == packet_sha256
            and decision.get("pair_sha256") == case["pair_sha256"], "Review packet/pair identity differs")
    for name in ("reviewer", "rationale"):
        require(isinstance(decision.get(name), str) and bool(decision[name].strip()), "Review requires " + name)
    reviewed = timestamp(decision.get("reviewed_at_utc"))
    require(earliest <= reviewed <= latest, "Review timestamp is outside packet/freeze interval")
    require(decision.get("condition_blind") is True and decision.get("condition_key_not_read") is True,
            "Reviewer must attest that condition assignment was hidden")
    values = [decision.get("A_correct"), decision.get("B_correct")]
    require(all(value is None or type(value) is bool for value in values)
            and "A_correct" in decision and "B_correct" in decision, "Correctness must be explicit true/false/null")
    category = decision.get("category")
    require(category in CATEGORIES, "Unknown review category")
    if category == "ambiguous_uncertain":
        require(None in values, "Ambiguous review must retain an uncertain judgment")
    else:
        require(all(type(value) is bool for value in values), "Uncertainty requires ambiguous category")
    if category == "both_substantively_valid":
        require(values == [True, True], "Both-valid category conflicts with correctness")
    if category == "both_substantively_wrong":
        require(values == [False, False], "Both-wrong category conflicts with correctness")
    if category == "content_difference":
        require(values[0] != values[1], "Content repair/harm requires different substantive correctness")
    if category == "formatting_only":
        require(values[0] == values[1], "Formatting-only category cannot assert a substantive repair")
    inspected = decision.get("inspected_images")
    require(isinstance(inspected, list) and len(inspected) == 2, "Both full page and region must be inspected")
    expected = {case[role]["path"]: case[role]["sha256"] for role in ("full_page", "evidence_region")}
    require(all(isinstance(item, dict) and set(item) == {"path", "sha256"} for item in inspected), "Unexpected inspected-image record")
    require(len({item["path"] for item in inspected}) == 2
            and {item["path"]: item["sha256"] for item in inspected} == expected, "Inspected image binding differs")


def freeze(packet_path: Path, review_paths: list[Path], output: Path):
    packet_path, output = Path(packet_path).resolve(), Path(output).resolve()
    require(not output.exists(), "Frozen review path already exists")
    packet = validate_packet(packet_path)
    packet_sha = sha(packet_path)
    frozen_at = now()
    entries, sources = [], []
    require(bool(review_paths), "At least one explicitly supplied reviewer ledger is required")
    for path in map(lambda p: Path(p).resolve(), review_paths):
        data = read(path)
        require(isinstance(data, dict) and data.get("packet_sha256") == packet_sha
                and isinstance(data.get("reviews"), list), "Reviewer ledger must bind packet and reviews")
        entries.extend(data["reviews"])
        sources.append({"path": str(path), "sha256": sha(path)})
    require(len({entry["case_id"] for entry in entries}) == len(entries), "Duplicate or conflicting case decisions require resolution")
    by_case = {entry["case_id"]: entry for entry in entries}
    require(set(by_case) == {case["case_id"] for case in packet["cases"]}, "Missing or unknown case decisions")
    ordered = []
    for case in packet["cases"]:
        decision = by_case[case["case_id"]]
        validate_decision(decision, case, packet_sha, timestamp(packet["created_utc"]), timestamp(frozen_at))
        ordered.append(decision)
    result = {"schema_version": 1, "status": "complete_blinded_review_frozen", "frozen_at_utc": frozen_at,
              "packet_sha256": packet_sha, "assignment_commitment_sha256": packet["assignment_commitment_sha256"],
              "source_ledgers": sources, "reviews": ordered, "reviewed_cases": len(ordered),
              "condition_key_read_by_freeze": False, "automatic_grades_changed": False,
              "limitations": "Blindness is a reviewer attestation plus a separate committed key, not independently observed behavior or third-party preregistration."}
    write_new(output, result)
    return {"status": result["status"], "reviews": len(ordered), "frozen_review_sha256": sha(output)}


def unblind(packet_path: Path, key_path: Path, frozen_path: Path, output: Path):
    packet_path, key_path, frozen_path, output = map(lambda p: Path(p).resolve(), (packet_path, key_path, frozen_path, output))
    require(not output.exists(), "Use a new unblinded report path")
    packet, key, frozen = validate_packet(packet_path), read(key_path), read(frozen_path)
    require(not key_path.is_relative_to(packet_path.parent), "Private key is inside public packet directory")
    packet_sha = sha(packet_path)
    require(frozen.get("status") == "complete_blinded_review_frozen"
            and frozen.get("automatic_grades_changed") is False and frozen.get("condition_key_read_by_freeze") is False,
            "A completed frozen blinded review is required")
    require(packet_sha == key["packet_sha256"] == frozen["packet_sha256"], "Packet binding differs")
    payload = {name: value for name, value in key.items() if name != "packet_sha256"}
    require(canonical_hash(payload) == packet["assignment_commitment_sha256"] == frozen["assignment_commitment_sha256"], "Condition assignment commitment differs")
    require(key["seed"] == SEED and key["role"] == "main", "Incorrect private-key study identity")
    require(timestamp(frozen["frozen_at_utc"]) <= timestamp(now()), "Frozen review timestamp is in the future")
    original_decisions = []
    for source in frozen["source_ledgers"]:
        require(sha(Path(source["path"])) == source["sha256"], "Original reviewer ledger changed after freeze")
        original = read(source["path"])
        require(original["packet_sha256"] == packet_sha, "Original reviewer packet binding differs")
        original_decisions.extend(original["reviews"])
    require(len({row["case_id"] for row in original_decisions}) == len(original_decisions), "Original reviewer ledgers overlap")
    original_map = {row["case_id"]: row for row in original_decisions}
    require(len(original_decisions) == len(frozen["reviews"])
            and all(original_map.get(row["case_id"]) == row for row in frozen["reviews"]), "Frozen decisions differ from original reviewer ledgers")
    require([row["case_id"] for row in key["keys"]] == [row["case_id"] for row in packet["cases"]]
            == [row["case_id"] for row in frozen["reviews"]], "Key/review/packet case coverage differs")
    require(frozen["reviewed_cases"] == len(packet["cases"]), "Frozen review count differs")
    require(len({row["source_cluster_id"] for row in key["keys"]}) == len(key["keys"]), "Repeated source cluster in key")
    repairs, harms, case_results = [], [], []
    for case, mapping, decision in zip(packet["cases"], key["keys"], frozen["reviews"]):
        validate_decision(decision, case, packet_sha, timestamp(packet["created_utc"]), timestamp(frozen["frozen_at_utc"]))
        require(mapping["pair_sha256"] == case["pair_sha256"]
                and (mapping["A"], mapping["B"]) == assignment(mapping["example_id"]), "Deterministic assignment differs")
        judgments = {mapping["A"]: decision["A_correct"], mapping["B"]: decision["B_correct"]}
        repair = decision["category"] == "content_difference" and judgments["native_256"] is True and judgments["degraded_256"] is False
        harm = decision["category"] == "content_difference" and judgments["native_256"] is False and judgments["degraded_256"] is True
        if repair:
            repairs.append(mapping["source_cluster_id"])
        if harm:
            harms.append(mapping["source_cluster_id"])
        case_results.append({"case_id": case["case_id"], "example_id": mapping["example_id"],
            "source_cluster_id": mapping["source_cluster_id"], "A": mapping["A"], "B": mapping["B"],
            "category": decision["category"], "native_correct_semantic": judgments["native_256"],
            "degraded_correct_semantic": judgments["degraded_256"], "native_only_content_repair": repair,
            "reverse_content_harm": harm, "rationale": decision["rationale"]})
    result = {"schema_version": 1, "created_utc": now(), "status": "descriptive_review_unblinded",
        "bindings": {"packet_sha256": packet_sha, "private_key_sha256": sha(key_path),
                     "frozen_review_sha256": sha(frozen_path), **key["bindings"]},
        "reviewed_cases": len(case_results), "category_counts": dict(Counter(row["category"] for row in case_results)),
        "native_only_content_repairs": len(repairs), "reverse_content_harms": len(harms),
        "net_content_repairs": len(repairs) - len(harms),
        "ambiguous_cases": sum(row["category"] == "ambiguous_uncertain" for row in case_results),
        "additional_descriptive_screen": {"at_least_three_distinct_unambiguous_native_only_repairs": len(set(repairs)) >= 3,
            "positive_net_repairs": len(repairs) > len(harms), "passes": len(set(repairs)) >= 3 and len(repairs) > len(harms),
            "cannot_rescue_failed_quantitative_rule": True, "authorizes_controller_training": False},
        "automatic_grades_changed": False, "model_generation_calls": 0, "cases": case_results,
        "limitations": ["All selected changes are reviewed; this is a descriptive subset, not a population accuracy estimate.",
                        "Judgments come from the supplied qualitative-review ledgers; they are not independent expert ground truth and do not change automatic grades.",
                        "Source/ROI privilege, ambiguity and formatting effects remain; no internal mechanism or learned policy is established."]}
    write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for flag in ("manifest", "lock", "run", "output", "key"):
        prepare_parser.add_argument("--" + flag, type=Path, required=True)
    freeze_parser = commands.add_parser("freeze")
    for flag in ("packet", "output"):
        freeze_parser.add_argument("--" + flag, type=Path, required=True)
    freeze_parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    unblind_parser = commands.add_parser("unblind")
    for flag in ("packet", "key", "frozen", "output"):
        unblind_parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.manifest, args.lock, args.run, args.output, args.key)
    elif args.command == "freeze":
        result = freeze(args.packet, args.reviews, args.output)
    else:
        result = unblind(args.packet, args.key, args.frozen, args.output)
        result = {name: value for name, value in result.items() if name != "cases"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
