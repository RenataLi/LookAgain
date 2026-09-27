"""Source-cluster selection and finalization guards without PDFs or inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/prepare_dude_replication.py"
SPEC = importlib.util.spec_from_file_location("dude_selection_test_module", PATH)
dude = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dude)


def source_fixture(doc_ids, *, no_lineage=(), no_match=()):
    docs, matches, assets = {}, {}, {}
    for i, doc in enumerate(doc_ids):
        docs[doc] = {
            "source_urls": [] if doc in no_lineage else [f"https://example.org/{doc}.pdf"],
            "lineage_verified": doc not in no_lineage, "prior_source_matches": [],
            "source_pdf_sha256": str(i) * 64, "source_ocr_sha256": "1" * 64,
            "source_azure_original_sha256": "2" * 64,
        }
        assets[(doc, "pdf")] = {"relative_path": f"pdf/{doc}.pdf"}
        if doc not in no_match:
            matches[doc] = {"question_uid": f"question-{i}", "doc_id": doc,
                            "canonical_match": {"support": [{"full_coverage": True}]}}
    return docs, matches, dude.DisjointSources(doc_ids), assets


def independent_order(text):
    return hashlib.sha256(("20260927:" + text).encode()).hexdigest(), text


def test_cluster_selects_global_first_question_not_first_document():
    ids = ["a" * 32, "b" * 32]
    docs, matches, clusters, assets = source_fixture(ids)
    docs_by_hash = sorted(ids, key=independent_order)
    question_names = sorted(["question-alpha", "question-beta"], key=independent_order)
    matches[docs_by_hash[0]]["question_uid"] = question_names[1]
    matches[docs_by_hash[1]]["question_uid"] = question_names[0]
    clusters.join(ids[0], ids[1], "same-source")
    result = dude.assemble_cluster_candidates(docs, matches, clusters, assets)
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["doc_id"] == docs_by_hash[1]
    assert result["candidates"][0]["question_uid"] == question_names[0]


def test_lineage_missing_main_is_excluded_but_probe_cluster_stays_reserved_without_candidate():
    ids = [f"{i:032x}" for i in range(1, 14)] + [dude.PROBE_DOC_ID]
    docs, matches, clusters, assets = source_fixture(
        ids, no_lineage=(ids[0], dude.PROBE_DOC_ID), no_match=(dude.PROBE_DOC_ID,))
    result = dude.assemble_cluster_candidates(docs, matches, clusters, assets)
    assert all(row["doc_id"] != ids[0] for row in result["candidates"])
    assert len(result["new_candidates"]) == 12
    assert len(result["reserved_ids"]) == 11  # Ten new sources plus the whole known probe cluster.
    assert sum(row["cohort_reservation"] == "engineering" for row in result["candidates"]) == 10
    probe_cluster = next(row for row in result["clusters"] if dude.PROBE_DOC_ID in row["doc_ids"])
    assert probe_cluster["source_cluster_id"] in result["reserved_ids"]
    assert probe_cluster["exclusion_reasons"] == ["no_eligible_question_with_verified_lineage"]


def test_historical_overlap_excludes_entire_connected_cluster():
    ids = ["a" * 32, "b" * 32]
    docs, matches, clusters, assets = source_fixture(ids)
    docs[ids[0]]["prior_source_matches"] = ["old_2019.pdf"]
    clusters.join(*ids, "exact-pdf")
    result = dude.assemble_cluster_candidates(docs, matches, clusters, assets)
    assert result["candidates"] == []
    assert result["clusters"][0]["exclusion_reasons"] == ["prior_tatdqa_source_overlap"]


def test_source_reservations_and_candidate_order_are_deterministic():
    ids = [f"{i:032x}" for i in range(1, 14)]
    a = source_fixture(ids)
    b = source_fixture(list(reversed(ids)))
    # Keep input candidates identical while varying dictionary/insertion order.
    b = (a[0], copy.deepcopy(a[1]), b[2], a[3])
    first = dude.assemble_cluster_candidates(*a)
    second = dude.assemble_cluster_candidates(*b)
    assert [r["source_cluster_id"] for r in first["candidates"]] == [r["source_cluster_id"] for r in second["candidates"]]
    expected = {r["source_cluster_id"] for r in first["candidates"][:10]}
    assert first["reserved_ids"] == expected == second["reserved_ids"]


def test_bottom64_flags_are_review_only_and_exact_clusters_are_not_reflagged():
    docs = {
        "a": {"bottom64_five_word_shingle_sketch": [str(x) for x in range(64)]},
        "b": {"bottom64_five_word_shingle_sketch": [str(x) for x in range(16, 80)]},
        "c": {"bottom64_five_word_shingle_sketch": [str(x) for x in range(17, 81)]},
    }
    clusters = dude.DisjointSources(docs)
    flags = dude.near_duplicate_flags(docs, clusters)
    assert next(f for f in flags if f["left_doc_id"] == "a" and f["right_doc_id"] == "b")["shared_bottom64_shingles"] == 48
    assert not any(f["left_doc_id"] == "a" and f["right_doc_id"] == "c" for f in flags)
    assert all(f["automatic_merge"] is False for f in flags)
    clusters.join("a", "b", "exact-url")
    assert not any(f["left_doc_id"] == "a" and f["right_doc_id"] == "b" for f in dude.near_duplicate_flags(docs, clusters))


def test_prospective_main_size_cannot_be_lowered(tmp_path):
    with pytest.raises(ValueError, match="660"):
        dude.finalize(tmp_path / "missing", [], tmp_path / "missing-review", tmp_path / "out", main_count=659)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["example_id", "source_cluster_id", "arbitrary_reason", "wrong_error_type"])
def test_deterministic_render_skip_cannot_change_rank_identity_or_invent_exclusions(mutation):
    candidate = {"example_id": "dude:q", "source_cluster_id": "source:a"}
    failure = {
        **candidate, "deterministic_source_exclusion": True,
        "infrastructure_resolution_required": False, "error_type": "Ineligible",
        "reason": "actual_raster_dimensions_differ_from_verified_geometry",
    }
    dude.validate_render_failure(failure, candidate)
    if mutation in {"example_id", "source_cluster_id"}:
        failure[mutation] = "other"
    elif mutation == "arbitrary_reason":
        failure["reason"] = "did_not_like_question"
    else:
        failure["error_type"] = "TimeoutExpired"
    with pytest.raises(ValueError):
        dude.validate_render_failure(failure, candidate)



def test_insufficient_reviewed_sources_do_not_write_a_smaller_final_panel(tmp_path):
    census_path, reviews = tmp_path / "census.json", tmp_path / "reviews.jsonl"
    census_path.write_text(json.dumps({
        "bindings": {"adapter_sha256": dude.sha256_file(PATH)},
        "stage": "source_only_technical_census", "seed": 20260927,
        "model_outputs_used": False, "final_selection_frozen": False,
        "extraction": {"verified_complete": True}, "candidates": [],
    }), encoding="utf-8")
    reviews.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Only0 source-reviewed"):
        dude.finalize(census_path, [], reviews, tmp_path / "out")
    assert not (tmp_path / "out").exists()
