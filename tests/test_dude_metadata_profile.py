"""DUDE metadata checks without PDFs, inference or dataset downloads."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/profile_dude_metadata.py"
SPEC = importlib.util.spec_from_file_location("dude_profile_test_module", PATH)
dude = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dude)


def box(**kwargs):
    return {"left": 158, "top": 214, "width": 1821, "height": 29, "page": 0, **kwargs}


def test_large_integer_rectangles_are_not_assumed_to_use_1000_unit_canvas():
    result = dude.inspect_boxes([[box()]])
    assert result["valid"] is True
    assert result["union_xyxy_if_same_page"] == [158, 214, 1979, 243]


def test_multiple_answer_pages_are_not_silently_merged():
    result = dude.inspect_boxes([[box(), box(page=8)]])
    assert result["valid"] is True
    assert result["same_annotated_page"] is False
    assert result["pages"] == [0, 8]
    assert result["union_xyxy_if_same_page"] is None


@pytest.mark.parametrize("change", [{"left": -1}, {"width": 0}, {"page": -1},
                                   {"height": float("nan")}, {"top": 0.5},
                                   {"page": False}])
def test_invalid_box_coordinates_fail_structure_checks(change):
    assert dude.inspect_boxes([[box(**change)]])["valid"] is False


@pytest.mark.parametrize("value", [None, [], [[], []], [[]], [[{"left": 1}]]])
def test_invalid_nested_shape_is_not_an_roi_candidate(value):
    assert dude.inspect_boxes(value)["valid"] is False


def test_url_normalization_preserves_document_distinctions():
    a = "https://www.Example.org/Report%20A.pdf?x=1&utm_source=search#page=2"
    assert dude.normalized_url(a) == "example.org/Report A.pdf?x=1"
    assert dude.normalized_url(a) != dude.normalized_url("https://example.org/report%20a.pdf?x=1")
    assert dude.normalized_url(a) != dude.normalized_url("https://example.org/Report%20A.pdf?x=2")
    assert dude.normalized_url("file:///local.pdf") is None


def test_single_short_answer_screen_retains_original_reference():
    row = {"answer_type": "extractive", "answers": ["$ 5 million"]}
    assert dude.short_extractive_failure(row) is None
    assert row["answers"] == ["$ 5 million"]
    assert dude.short_extractive_failure({"answer_type": "list/extractive", "answers": ["a"]}) == "not_extractive"
    assert dude.short_extractive_failure({"answer_type": "extractive", "answers": ["a", "b"]}) == "not_one_answer"
    assert dude.short_extractive_failure({"answer_type": "extractive", "answers": [" " ]}) == "not_nonempty_string"


def test_empirical_quantiles_have_independent_expected_values():
    assert dude.quantiles([0, 10, 20, 30]) == pytest.approx({
        "min": 0, "p25": 7.5, "median": 15.0, "p75": 22.5, "p95": 28.5, "max": 30})


def synthetic_profile_inputs(tmp_path, monkeypatch, *, conflicting=False):
    import csv
    import hashlib
    import json

    extractive = {
        "questionId": "extractive", "question": "Which value?", "answers": ["five"],
        "answer_type": "extractive", "docId": "a" * 32, "data_split": "train",
        "answers_page_bounding_boxes": [[box()]],
    }
    repeated = {
        "questionId": "duplicate", "question": "Explain the result.", "answers": ["explanation"],
        "answer_type": "abstractive", "docId": "b" * 32, "data_split": "train",
    }
    other_copy = dict(repeated)
    if conflicting:
        other_copy["answers"] = ["different explanation"]
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        "dataset_name": "fixture", "dataset_version": "fixture",
        "data": [extractive, repeated, other_copy,
                 {"data_split": "val"}, {"data_split": "test"}],
    }), encoding="utf-8")
    lineage = tmp_path / "lineage.csv"
    with lineage.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "url", "url_md5 (filename)", "master_hash", "licenseurl", "stage", "license", "found_url"])
        writer.writeheader()
        writer.writerow({"url": "https://example.org/source.pdf",
                         "url_md5 (filename)": "a" * 32})
    prior_main, prior_smoke = tmp_path / "main.jsonl", tmp_path / "smoke.jsonl"
    prior_main.write_text("".join(json.dumps({"source_id": f"Prior{i}_2019.pdf"}) + "\n"
                                 for i in range(100)), encoding="utf-8")
    prior_smoke.write_text("".join(json.dumps({"source_id": f"Prior{i}_2019.pdf"}) + "\n"
                                  for i in range(100, 102)), encoding="utf-8")
    paths = {"annotations": metadata, "lineage": lineage,
             "prior_main": prior_main, "prior_smoke": prior_smoke}
    monkeypatch.setattr(dude, "PINS", {
        key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()})
    return metadata, lineage, prior_main, prior_smoke, tmp_path / "output"


def test_identical_duplicate_audit_preserves_raw_counts_and_excludes_other_splits(tmp_path, monkeypatch):
    result = dude.profile(*synthetic_profile_inputs(tmp_path, monkeypatch))
    train = result["training"]
    assert train["questions"] == 3
    assert train["unique_question_ids"] == 2
    assert train["short_single_extractive_questions"] == 1
    assert train["same_page_metadata_candidate_questions"] == 1
    assert train["metadata_first_failure_counts"] == {"not_extractive": 2}
    assert train["duplicate_question_groups"] == [{
        "question_id": "duplicate", "copies": 2, "answer_type": "abstractive", "identical": True}]
    # These rows lack every eligibility/box field and would fail if inspected.
    assert result["split_policy"]["split_row_inventory_only"] == {"train": 3, "val": 1, "test": 1}
    assert result["split_policy"]["validation_test_eligibility_or_bbox_inspected"] is False


def test_conflicting_duplicate_question_id_fails_before_output(tmp_path, monkeypatch):
    paths = synthetic_profile_inputs(tmp_path, monkeypatch, conflicting=True)
    with pytest.raises(ValueError, match="Conflicting duplicate training questionId"):
        dude.profile(*paths)
    assert not paths[-1].exists()
