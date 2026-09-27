"""Independent checks for source-census/audit/manifest/review agreement."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "dude_preparation_binding_test", Path(__file__).resolve().parents[1] / "experiments/prepare_dude_replication.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


@pytest.fixture
def bound():
    candidate = {
        "example_id": "dude:q", "source_id": "cluster", "source_cluster_id": "cluster",
        "source_cluster_doc_ids": ["doc"], "doc_id": "doc", "question_uid": "q",
        "question": "Which amount?", "answer": "$10", "original_answers": ["$10"],
        "original_answer_variants": ["$10.00", "ten dollars"], "ocr_supported_answers": ["$10", "$10.00"],
        "answer_page_index": 2, "answer_page_privileged": True, "roi_privileged": True,
        "roi_pixels": [100, 100, 400, 400], "pixel_dimensions": [1700, 2200], "pdf_page_count": 5,
        "source_pdf_sha256": "pdf", "source_ocr_sha256": "due", "source_azure_original_sha256": "azure",
        "source_rank": 12, "lineage_verified": True, "source_urls": ["https://example.org/report.pdf"],
        "prior_engineering_cluster": False, "near_duplicate_review_flags": [],
        "canonical_match": {"support": [{"full_coverage": True}]}, "cohort_reservation": "main_candidate",
    }
    candidate["technical_provenance_sha256"] = adapter.canonical_hash(candidate)
    audit = {**copy.deepcopy(candidate), "source_image_path": "images/doc.png", "image_sha256": "image",
             "actual_render_size": [1700, 2200]}
    row = {**copy.deepcopy(candidate), "validated_primary_answers": [], "source_semantic_audit_status": "pending",
           "cohort": "main_candidate", "image_path": "images/doc.png", "image_sha256": "image",
           "rendered_png_sha256": "image", "width": 1700, "height": 2200,
           "partial_gold_word_coverage": False, "roi_provenance_sha256": adapter.canonical_hash(audit)}
    review = {"example_id": "dude:q", "source_cluster_id": "cluster",
              "source_audit_sha256": row["roi_provenance_sha256"], "semantic_approved": True,
              "reviewer": "synthetic test reviewer", "reviewed_at_utc": "2026-09-26T00:00:00Z",
              "rationale": "Synthetic fixture; visible amount and referent verified.", "model_outputs_used": False,
              "full_page_visual_check": True, "roi_context_sufficient": True, "duplicate_identity_resolved": True,
              "validated_primary_answers": ["$10", "$10.00"]}
    return candidate, audit, row, review


def test_consistent_source_artifacts_and_approved_original_variants_pass(bound):
    candidate, audit, row, review = bound
    adapter.validate_candidate_binding(row, audit, candidate)
    adapter.validate_source_review(row, review)


@pytest.mark.parametrize("field,value", [
    ("question", "A different question?"), ("answer", "$100"), ("original_answers", ["$100"]),
    ("original_answer_variants", ["a newly invented variant"]), ("ocr_supported_answers", ["$100"]),
    ("source_cluster_id", "other_cluster"), ("source_id", "other_cluster"),
    ("source_cluster_doc_ids", ["other_doc"]), ("source_rank", 1000),
    ("roi_pixels", [200, 200, 500, 500]), ("answer_page_index", 0),
    ("source_pdf_sha256", "different_pdf"), ("answer_page_privileged", False),
])
def test_manifest_edit_cannot_retain_authoritative_source_audit(field, value, bound):
    candidate, audit, row, _ = bound
    row[field] = value
    # File/digest consistency alone would not catch this field disagreement.
    with pytest.raises(ValueError, match=field):
        adapter.validate_candidate_binding(row, audit, candidate)


def test_matching_manifest_and_audit_edit_still_disagrees_with_census(bound):
    candidate, audit, row, _ = bound
    audit["question"] = row["question"] = "Edited question?"
    row["roi_provenance_sha256"] = adapter.canonical_hash(audit)
    with pytest.raises(ValueError, match="question"):
        adapter.validate_candidate_binding(row, audit, candidate)


def test_census_content_change_requires_updated_technical_provenance(bound):
    candidate, audit, row, _ = bound
    for data in (candidate, audit, row):
        data["source_rank"] = 13
    with pytest.raises(ValueError, match="Technical candidate provenance"):
        adapter.validate_candidate_binding(row, audit, candidate)


@pytest.mark.parametrize("field,value", [
    ("validated_primary_answers", ["$10"]), ("source_semantic_audit_status", "approved"),
    ("cohort", "engineering"), ("image_sha256", "anotherimage"), ("width", 1600),
    ("partial_gold_word_coverage", True),
])
def test_candidate_cannot_silently_become_approved_or_change_render_metadata(field, value, bound):
    candidate, audit, row, _ = bound
    row[field] = value
    with pytest.raises(ValueError):
        adapter.validate_candidate_binding(row, audit, candidate)


def test_rejection_is_audited_without_being_forced_to_approve_visual_content(bound):
    _, _, row, review = bound
    review = {key: value for key, value in review.items() if key not in
              ("full_page_visual_check", "roi_context_sufficient", "duplicate_identity_resolved", "validated_primary_answers")}
    review.update(semantic_approved=False, rationale="Canonical source reference is unsupported; exclude cluster.")
    adapter.validate_source_review(row, review)


@pytest.mark.parametrize("decision", [False, True])
@pytest.mark.parametrize("field", ["reviewer", "reviewed_at_utc", "rationale", "model_outputs_used"])
def test_approval_and_rejection_both_require_source_only_review_evidence(decision, field, bound):
    _, _, row, review = bound
    review["semantic_approved"] = decision
    review.pop(field)
    with pytest.raises(ValueError):
        adapter.validate_source_review(row, review)


@pytest.mark.parametrize("timestamp", ["2026-09-26T00:00:00", "tomorrow", "2026-15-99T00:00:00Z"])
def test_review_timestamp_must_be_valid_and_timezone_aware(timestamp, bound):
    _, _, row, review = bound
    review["reviewed_at_utc"] = timestamp
    with pytest.raises(ValueError, match="timezone-aware"):
        adapter.validate_source_review(row, review)


def test_review_is_bound_to_exact_source_audit(bound):
    _, _, row, review = bound
    review["source_audit_sha256"] = "other_source_audit"
    with pytest.raises(ValueError, match="bound"):
        adapter.validate_source_review(row, review)


@pytest.mark.parametrize("validated", [["$10.00"], ["$10", "ten dollars"], ["$10", "10 dollars"]])
def test_semantic_approval_cannot_replace_canonical_or_add_unsupported_variants(validated, bound):
    _, _, row, review = bound
    review["validated_primary_answers"] = validated
    with pytest.raises(ValueError, match="retain canonical"):
        adapter.validate_source_review(row, review)


def test_partial_word_coverage_requires_its_own_review_even_if_full_page_was_seen(bound):
    _, _, row, review = bound
    row["partial_gold_word_coverage"] = True
    with pytest.raises(ValueError, match="Partial OCR-word"):
        adapter.validate_source_review(row, review)
    review["partial_word_coverage_checked"] = True
    adapter.validate_source_review(row, review)
