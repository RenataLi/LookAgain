"""Source identity and census-integrity checks; no corpus, PDF parser or model."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/profile_replication_sources.py"
SPEC = importlib.util.spec_from_file_location("replication_profile_test_module", PATH)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


def test_report_identity_excludes_aliases_and_all_pages_of_old_source():
    sources = {"Example-Inc_2019.pdf", "EXAMPLE_INC_2019.PDF",
               "example-inc_2020.pdf", "Different_2019.pdf"}
    remaining = profile.remaining_sources(sources, {"Example-Inc_2019.pdf"})
    assert remaining == {"example-inc_2020.pdf", "Different_2019.pdf"}
    assert profile.company_key("Example-Inc_2019.pdf") == "example"


@pytest.mark.parametrize("name", ["report.pdf", "../report_2019.pdf",
                                  "dir\\report_2019.pdf", "_2019.pdf", "x_2019.txt"])
def test_unknown_or_unsafe_report_identity_fails_closed(name):
    with pytest.raises(ValueError):
        profile.report_key(name)


def test_missing_historical_report_is_not_silently_ignored():
    with pytest.raises(ValueError, match="Historical report"):
        profile.remaining_sources({"New_2019.pdf"}, {"Old_2019.pdf"})


def test_generic_table_flag_is_a_specific_question_screen():
    assert profile.semantic_flags("What does the table show?") == ["generic_table_referent"]
    assert profile.semantic_flags("  What do these tables contain?") == ["generic_table_referent"]
    assert profile.semantic_flags("What was revenue in 2019 in the table?") == []
    assert profile.semantic_flags("How much were employee costs between 2017 and 2018?") == [
        "between_years_level_vs_change_review"]


def census_fixture():
    questions = [
        {"source_id": "New_2019.pdf", "question_uid": "a", "eligible": True,
         "semantic_review_flags": []},
        {"source_id": "New_2019.pdf", "question_uid": "b", "eligible": True,
         "semantic_review_flags": ["generic_table_referent"]},
        {"source_id": "Other_2019.pdf", "question_uid": "c", "eligible": False,
         "first_exclusion_reason": "missing_mapping"},
    ]
    result = {
        "prior_exclusion": {"source_ids": ["Old_2019.pdf"]},
        "remaining": {
            "source_profiles": [
                {"source_id": "New_2019.pdf", "questions": 2, "strict_eligible_questions": 2},
                {"source_id": "Other_2019.pdf", "questions": 1, "strict_eligible_questions": 0},
            ],
            "source_upper_bound_before_filters": 2, "remaining_questions": 3,
            "strict_eligible_questions": 2, "strict_eligible_reports": 1,
            "strict_eligible_questions_if_generic_table_questions_removed": 1,
            "strict_eligible_reports_if_generic_table_questions_removed": 1,
            "question_first_failure_counts": {"missing_mapping": 1},
        },
    }
    return result, questions


def test_multiple_questions_are_counted_as_one_report_and_flags_are_separate():
    result, questions = census_fixture()
    profile.validate_census(result, questions)


@pytest.mark.parametrize("mutation", ["duplicate_question", "old_source", "wrong_report_count",
                                      "missing_failure", "wrong_filtered_count", "missing_profile"])
def test_census_rejects_identity_and_denominator_corruption(mutation):
    result, questions = census_fixture()
    if mutation == "duplicate_question":
        questions[1]["question_uid"] = "a"
    elif mutation == "old_source":
        result["prior_exclusion"]["source_ids"].append("New_2019.pdf")
    elif mutation == "wrong_report_count":
        result["remaining"]["strict_eligible_reports"] = 2
    elif mutation == "missing_failure":
        result["remaining"]["question_first_failure_counts"] = {}
    elif mutation == "wrong_filtered_count":
        result["remaining"]["strict_eligible_questions_if_generic_table_questions_removed"] = 2
    else:
        questions[0]["source_id"] = "Absent_2019.pdf"
    with pytest.raises(ValueError):
        profile.validate_census(result, questions)


def test_bound_file_hash_rejects_changed_bytes(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"abc")
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert profile.verify_pin(path, digest) == digest
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="SHA256 pin"):
        profile.verify_pin(path, digest)


def test_historical_manifest_must_match_raw_question_and_report():
    entries, rows = [], []
    for i in range(102):
        entries.append({"doc": {"uid": str(i), "source": f"Report{i}_2019.pdf"},
                        "questions": [{"uid": str(i), "question": "What?", "answer": ["value"]}]})
        rows.append({"doc_uid": str(i), "source_id": f"Report{i}_2019.pdf",
                     "question_uid": str(i), "question": "What?", "answer": "value"})
    profile.validate_historical(rows, entries)
    corrupted = copy.deepcopy(rows)
    corrupted[0]["source_id"] = "Different_2019.pdf"
    with pytest.raises(ValueError, match="raw annotation"):
        profile.validate_historical(corrupted, entries)


def test_nonempty_output_rejected_before_loading_raw_or_pdf_dependencies():
    with pytest.raises(FileExistsError, match="never overwritten"):
        profile.profile(Path("missing_raw"), Path("missing_main"),
                        Path("missing_smoke"), PATH.parent)

