"""Source-only geometry and text-boundary tests; no model or corpus required."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/prepare_evidence_roi.py"
SPEC = importlib.util.spec_from_file_location("evidence_roi_test_module", PATH)
roi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roi)


def example(text="Net $5 million.", offsets=None):
    words = text.split()
    offsets = [4, 14] if offsets is None else offsets
    block = {"uuid": "b", "text": text, "bbox": [10, 10, 400, 30],
        "words": {"word_list": words, "bbox_list": [[10 + i * 100, 10, 90 + i * 100, 30] for i in range(len(words))]}}
    return {"block_mapping": [{"b": offsets}]}, {"pages": [{"bbox": [0, 0, 1000, 1000], "blocks": [block]}]}


def geometry(media=None, crop=None):
    return {"page_count": 1, "mediabox": media or [0, 0, 1000, 2000],
            "cropbox": crop or [100, 100, 900, 1900], "rotation": 0, "user_unit": 1}


def test_monotonic_word_alignment_keeps_original_character_offsets():
    assert roi.align_words(" one\t two   one ", ["one", "two", "one"]) == [[1, 4], [6, 9], [12, 15]]
    assert roi.normalize_span("  USD\t5\n MILLION ") == "usd 5 million"
    with pytest.raises(roi.Ineligible, match="alignment"):
        roi.align_words("one EXTRA two", ["one", "two"])


def test_deduplicate_exact_mapping_triples_without_dropping_distinct_evidence():
    question, ocr = example()
    question["block_mapping"] *= 2
    result = roi.evidence_words(question, ocr)
    assert result["selected_text"] == "$5 million"
    assert result["duplicate_mappings_removed"] == 1
    assert len(result["mappings"]) == 1 and len(result["selected_words"]) == 2
    assert result["unsafe_boundary_cut"] is False  # Terminal period does not invalidate a word box.


@pytest.mark.parametrize("text,offsets,unsafe", [
    ("17.88 years", [1, 11], True), ("two-thirds", [4, 9], True),
    ("$5 million,", [0, 10], False), ("(in thousands)", [1, 13], False),
    ("foo-bar", [4, 7], True), ("tax", [0, 2], True)])
def test_partial_word_annotations_cannot_silently_supply_wrong_evidence(text, offsets, unsafe):
    question, ocr = example(text, offsets)
    assert roi.evidence_words(question, ocr)["unsafe_boundary_cut"] is unsafe


@pytest.mark.parametrize("mutation,expected", [
    ("missing", "missing_mapping"), ("offsets", "invalid_offsets"),
    ("boxes", "word_box_count_mismatch"), ("outside", "outside_page"),
    ("crosspage", "non_single_page")])
def test_invalid_annotation_geometry_fails_closed(mutation, expected):
    question, ocr = example()
    if mutation == "missing":
        question["block_mapping"] = []
    elif mutation == "offsets":
        question["block_mapping"] = [{"b": [3, 9000]}]
    elif mutation == "boxes":
        ocr["pages"][0]["blocks"][0]["words"]["bbox_list"].pop()
    elif mutation == "outside":
        ocr["pages"][0]["blocks"][0]["words"]["bbox_list"][1] = [999, 10, 1100, 30]
    else:
        ocr["pages"].append(copy.deepcopy(ocr["pages"][0]))
    with pytest.raises(roi.Ineligible, match=expected):
        roi.evidence_words(question, ocr)


def test_cropbox_translation_and_pdf_y_axis_are_not_naive_fullpage_scaling():
    result, transform = roi.map_ocr_box([100, 200, 200, 300], [0, 0, 800, 1800], geometry(), [2000, 4000])
    assert result == [400, 600, 600, 800]
    assert transform["offset_xy"] == [200, 200]
    assert transform["scale_xy"] == [2, 2]
    shifted = geometry([10, 20, 1010, 2020], [110, 120, 910, 1920])
    assert roi.map_ocr_box([100, 200, 200, 300], [0, 0, 800, 1800], shifted, [2000, 4000])[0] == result


def test_equal_media_and_crop_boxes_reduce_to_axiswise_raster_scaling():
    g = geometry([0, 0, 1000, 1000], [0, 0, 1000, 1000])
    assert roi.map_ocr_box([10, 20, 21, 31], [0, 0, 1000, 1000], g, [1501, 1502])[0] == [15, 30, 32, 47]


@pytest.mark.parametrize("change,reason", [("rotation", "rotation"), ("unit", "user_unit"),
    ("crop", "outside_mediabox"), ("aspect", "aspect")])
def test_unsupported_pdf_frames_are_exclusions_not_guessed_fallbacks(change, reason):
    g = geometry()
    page = [0, 0, 800, 1800]
    if change == "rotation":
        g["rotation"] = 90
    elif change == "unit":
        g["user_unit"] = 2
    elif change == "crop":
        g["cropbox"] = [-10, -10, 1010, 2010]
    else:
        page[3] = 1600
    with pytest.raises(roi.Ineligible, match=reason):
        roi.map_ocr_box([100, 200, 200, 300], page, g, [2000, 4000])


def test_padded_window_is_minimum256_and_shifts_at_edge_without_blank_padding():
    g = geometry([0, 0, 1000, 1000], [0, 0, 1000, 1000])
    box, info = roi.padded_roi([[980, 985, 998, 999]], [0, 0, 1000, 1000], g, [1000, 1000])
    assert box == [744, 744, 1000, 1000]
    assert roi.contained([980, 985, 998, 999], box)
    assert info["context_margin_pixels"] == 28
    assert info["roi_page_area_fraction"] == 256 * 256 / 1_000_000


def test_margin_uses_median_word_height_and_large_context_fails_cap():
    g = geometry([0, 0, 1000, 1000], [0, 0, 1000, 1000])
    box, info = roi.padded_roi([[400, 400, 420, 410], [425, 400, 480, 430], [485, 400, 530, 420]], [0, 0, 1000, 1000], g, [1000, 1000])
    assert info["median_word_height_pixels"] == 20 and info["context_margin_pixels"] == 40
    assert box[2] - box[0] == box[3] - box[1] == 256
    with pytest.raises(roi.Ineligible, match="quarter_page"):
        roi.padded_roi([[0, 0, 800, 800]], [0, 0, 1000, 1000], g, [1000, 1000])


def test_provenance_hash_matches_exact_unicode_contract_and_detects_coordinates():
    record = {"example_id": "ex", "reference": "Company’s", "roi_pixels": [1, 2, 257, 258]}
    expected = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert roi.canonical_hash(record) == expected
    record["roi_pixels"][0] += 1
    assert roi.canonical_hash(record) != expected
