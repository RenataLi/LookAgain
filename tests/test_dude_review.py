"""Synthetic only: blind selection, commitments and descriptive review guards."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import prepare_dude_review as review


def record(parsed, raw=None, score=0):
    return {"predicted_answer": parsed, "parse_valid": parsed is not None,
            "response": raw if raw is not None else "ANSWER: " + str(parsed), "primary_em": score}


@pytest.mark.parametrize("left,right,selected", [
    (record("x", score=1), record("y", score=1), True),
    (record("x", score=0), record("y", score=0), True),
    (record("X"), record("x"), True),
    (record("x", "ANSWER: x"), record("x", "x"), False),
    (record(None, "unparsed1"), record(None, "unparsed2"), True),
    (record(None, "same"), record(None, "same"), False),
    (record(None, "x"), record("x", "x"), True),
])
def test_all_protocol_selection_edges(left, right, selected):
    assert review.changed_pair(left, right) is selected


def test_parse_schema_rejects_invalid_nonnull():
    bad = record("x")
    bad["parse_valid"] = False
    with pytest.raises(ValueError, match="Inconsistent"):
        review.changed_pair(bad, record("y"))


def make_packet(tmp_path, monkeypatch, n=5, unchanged=False):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "native_original_document_id.png"
    Image.new("RGB", (40, 50), (70, 90, 110)).save(image)
    rows, records = [], {}
    for i in range(n):
        row = {"example_id": f"document-{i}-question", "source_cluster_id": f"cluster-{i}",
               "image_path": image.name, "image_sha256": review.sha(image), "roi_pixels": [4, 5, 30, 40],
               "question": f"What is item {i}?", "original_answers": ["correct"],
               "original_answer_variants": ["CORRECT", "incorrect variant"],
               "validated_primary_answers": ["correct", "CORRECT"]}
        rows.append(row)
        records[row["example_id"], "native_256"] = record("answer A", score=0)
        records[row["example_id"], "degraded_256"] = record("answer A" if unchanged else "answer B", score=0)
    manifest = source / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    lock = tmp_path / "lock.json"
    lock.write_text("{}", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    for name in ("run.json", "records.jsonl", "completed.json"):
        (run / name).write_text("{}", encoding="utf-8")
    calls = []
    def validator(m, l, r):
        calls.append((m, l, r))
        return rows, records, {}, {}
    monkeypatch.setattr(review, "_validated_run", validator)
    output, key = tmp_path / "packet", tmp_path / "private" / "key.json"
    result = review.prepare(manifest, lock, run, output, key)
    assert calls == [(manifest.resolve(), lock.resolve(), run.resolve())]
    return output / "packet.json", key, result, rows, records


def make_ledger(packet_path, key_path, tmp_path, outcomes=None):
    packet, key = review.read(packet_path), review.read(key_path)
    decisions = []
    for index, (case, mapping) in enumerate(zip(packet["cases"], key["keys"])):
        # Test-only synthetic decisions. Production code never generates these.
        native, degraded, category = outcomes[index] if outcomes else (True, False, "content_difference")
        meanings = {"native_256": native, "degraded_256": degraded}
        decisions.append({"case_id": case["case_id"], "packet_sha256": review.sha(packet_path),
            "pair_sha256": case["pair_sha256"], "reviewer": "synthetic-test-only", "reviewed_at_utc": review.now(),
            "condition_blind": True, "condition_key_not_read": True,
            "A_correct": meanings[mapping["A"]], "B_correct": meanings[mapping["B"]],
            "category": category, "rationale": "Synthetic fixture: supplied judgments test the guard and aggregation only.",
            "inspected_images": [case["full_page"], case["evidence_region"]]})
    path = tmp_path / "reviewer.json"
    path.write_text(json.dumps({"packet_sha256": review.sha(packet_path), "reviews": decisions}), encoding="utf-8")
    return path


def test_packet_has_no_condition_score_source_or_original_filename_leak(tmp_path, monkeypatch):
    packet, key, result, rows, _ = make_packet(tmp_path, monkeypatch)
    data = review.read(packet)
    text = packet.read_text(encoding="utf-8")
    for forbidden in ("native_256", "degraded_256", "primary_em", "document-", "cluster-", "native_original_document_id"):
        assert forbidden not in text
    assert result["cases"] == 5
    assert key.parent != packet.parent
    expected_ids = [row["example_id"] for row in sorted(rows, key=lambda r: (review.case_order(r["example_id"]), r["example_id"]))]
    assert [row["example_id"] for row in review.read(key)["keys"]] == expected_ids
    assert "references_source_validated" in data["cases"][0]
    assert review.validate_packet(packet) == data


def test_assignment_stable_and_not_manifest_order_dependent():
    ids = [f"example{i}" for i in range(30)]
    first = {key: review.assignment(key) for key in ids}
    assert first == {key: review.assignment(key) for key in reversed(ids)}
    assert set(first.values()) == {review.PAIR_ACTIONS, tuple(reversed(review.PAIR_ACTIONS))}


def test_prepare_requires_complete_validator_and_external_key(tmp_path, monkeypatch):
    called = []
    def fail(*args):
        called.append(True)
        raise ValueError("incomplete run")
    monkeypatch.setattr(review, "_validated_run", fail)
    with pytest.raises(ValueError, match="incomplete"):
        review.prepare(tmp_path / "manifest", tmp_path / "lock", tmp_path / "run", tmp_path / "packet", tmp_path / "key")
    assert called and not (tmp_path / "packet").exists()
    with pytest.raises(ValueError, match="outside"):
        review.prepare(tmp_path / "manifest", tmp_path / "lock", tmp_path / "run", tmp_path / "packet", tmp_path / "packet" / "key.json")


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["reviews"].pop(), "Missing"),
    (lambda d: d["reviews"].append(deepcopy(d["reviews"][0])), "Duplicate"),
    (lambda d: d["reviews"][0].update(pair_sha256="0" * 64), "identity"),
    (lambda d: d["reviews"][0].update(condition_key_not_read=False), "hidden"),
    (lambda d: d["reviews"][0].update(reviewed_at_utc="2026-09-27T12:00:00"), "Naive"),
    (lambda d: d["reviews"][0].update(reviewed_at_utc="2000-01-01T00:00:00+00:00"), "interval"),
    (lambda d: d["reviews"][0].update(rationale=""), "rationale"),
    (lambda d: d["reviews"][0].update(A_correct=1), "true/false/null"),
    (lambda d: d["reviews"][0].update(A_correct=None), "Uncertainty"),
    (lambda d: d["reviews"][0].update(category="both_substantively_valid"), "conflicts"),
    (lambda d: d["reviews"][0].update(category="formatting_only"), "substantive repair"),
    (lambda d: d["reviews"][0].update(inspected_images=[]), "inspected"),
])
def test_freeze_rejects_missing_conflicting_or_unbound_decisions(tmp_path, monkeypatch, mutation, match):
    packet, key, _, _, _ = make_packet(tmp_path, monkeypatch)
    ledger = make_ledger(packet, key, tmp_path)
    data = review.read(ledger)
    mutation(data)
    ledger.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        review.freeze(packet, [ledger], tmp_path / "frozen.json")
    assert not (tmp_path / "frozen.json").exists()


def test_unblind_counts_three_repairs_one_harm_and_uncertainty(tmp_path, monkeypatch):
    packet, key, _, _, _ = make_packet(tmp_path, monkeypatch)
    ledger = make_ledger(packet, key, tmp_path, [(True, False, "content_difference")] * 3
                         + [(False, True, "content_difference"), (None, False, "ambiguous_uncertain")])
    frozen = tmp_path / "frozen.json"
    review.freeze(packet, [ledger], frozen)
    result = review.unblind(packet, key, frozen, tmp_path / "unblinded.json")
    assert result["native_only_content_repairs"] == 3
    assert result["reverse_content_harms"] == 1
    assert result["net_content_repairs"] == 2
    assert result["ambiguous_cases"] == 1
    assert result["additional_descriptive_screen"]["passes"] is True
    assert result["additional_descriptive_screen"]["cannot_rescue_failed_quantitative_rule"] is True
    assert result["automatic_grades_changed"] is False


def test_both_valid_wrong_and_formatting_are_not_repairs(tmp_path, monkeypatch):
    packet, key, _, _, _ = make_packet(tmp_path, monkeypatch, n=3)
    ledger = make_ledger(packet, key, tmp_path, [(True, True, "both_substantively_valid"),
                         (False, False, "both_substantively_wrong"), (True, True, "formatting_only")])
    frozen = tmp_path / "frozen.json"
    review.freeze(packet, [ledger], frozen)
    result = review.unblind(packet, key, frozen, tmp_path / "unblinded.json")
    assert result["native_only_content_repairs"] == result["reverse_content_harms"] == 0
    assert not result["additional_descriptive_screen"]["passes"]


@pytest.mark.parametrize("target", ["packet", "image", "key", "ledger", "frozen"])
def test_unblind_rejects_post_freeze_mutation(tmp_path, monkeypatch, target):
    packet, key, _, _, _ = make_packet(tmp_path, monkeypatch)
    ledger = make_ledger(packet, key, tmp_path)
    frozen = tmp_path / "frozen.json"
    review.freeze(packet, [ledger], frozen)
    if target == "image":
        path = packet.parent / review.read(packet)["cases"][0]["full_page"]["path"]
        path.write_bytes(b"modified image")
    else:
        path = {"packet": packet, "key": key, "ledger": ledger, "frozen": frozen}[target]
        data = review.read(path)
        if target == "packet":
            data["cases"][0]["raw_response_A"] = "changed"
        elif target == "key":
            data["keys"][0]["A"] = "changed"
        else:
            data["reviews"][0]["rationale"] = "changed after freeze"
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        review.unblind(packet, key, frozen, tmp_path / "unblinded.json")


def test_zero_changes_can_be_explicitly_reviewed_and_frozen(tmp_path, monkeypatch):
    packet, key, result, _, _ = make_packet(tmp_path, monkeypatch, unchanged=True)
    assert result["cases"] == 0
    ledger = make_ledger(packet, key, tmp_path)
    frozen = tmp_path / "frozen.json"
    review.freeze(packet, [ledger], frozen)
    unblinded = review.unblind(packet, key, frozen, tmp_path / "unblinded.json")
    assert unblinded["reviewed_cases"] == 0
    assert not unblinded["additional_descriptive_screen"]["passes"]


def test_local_asset_paths_fail_closed(tmp_path):
    for value in ("../image.png", "C:\\private\\image.png", "\\\\server\\share\\image.png"):
        with pytest.raises(ValueError, match="Unsafe"):
            review.relative_file(tmp_path, value)
