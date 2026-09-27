"""Independent source-only DUDE tests with hand-calculated geometry.

No corpus files, model, GPU, or experiment outcomes are needed. The coordinate
examples were specified in work/dude_adversarial_test_matrix.md before reading
the adapter; the base page height is 432pt to satisfy the locked source floor.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("prepare_dude_replication.py", "dude_independent_geometry")
metadata = load_module("profile_dude_metadata.py", "dude_independent_metadata")
DOC_ID = "a" * 32


def physical_page():
    # 10x6 inches at 200 DPI = 2000x1200; PDF coordinates are points.
    return {"page_count": 1, "mediabox": [0, 0, 720, 432],
            "cropbox": [0, 0, 720, 432], "rotation": 0, "user_unit": 1}


def azure_page():
    return {"page": 1, "width": 10, "height": 6, "unit": "inch", "angle": 0}


def assets(texts=("Net", "$5", "million"), boxes=None):
    if boxes is None:
        boxes = [[100 + 60 * i, 100, 140 + 60 * i, 120] for i in range(len(texts))]
    due = {"doc_id": DOC_ID + ".pdf", "tokens": list(texts),
           "positions": copy.deepcopy(boxes), "scores": [1.0] * len(texts),
           "structures": {"pages": {"structure_value": [[0, len(texts)]],
                                      "positions": [[0, 0, 2000, 1200]]}}}
    raw = azure_page()
    words = []
    for text, (left, top, right, bottom) in zip(texts, boxes):
        # Independent raw Azure inches: each coordinate divided by 200.
        words.append({"text": text, "boundingBox": [left / 200, top / 200,
                       right / 200, top / 200, right / 200, bottom / 200,
                       left / 200, bottom / 200]})
    raw["lines"] = [{"words": words}]
    original = {"analyzeResult": {"readResults": [raw]}}
    geometry = adapter.validate_page_geometry(physical_page(), raw,
                                              [0, 0, 2000, 1200], 0)
    return due, original, geometry


def question(answer="$5 million"):
    return {"questionId": DOC_ID + "_q", "docId": DOC_ID, "data_split": "train",
            "question": "What is the stated amount?", "answer_type": "extractive",
            "answers": [answer], "answers_variants": [],
            "answers_page_bounding_boxes": [[{"left": 90, "top": 90,
                       "width": 240, "height": 50, "page": 0}]]}


def assess(q, due=None, original=None):
    if due is None:
        due, original, _ = assets()
    return adapter.assess_question(q, DOC_ID, [physical_page()], due, original, metadata)


def words(texts, boxes=None):
    boxes = boxes or [[100 + i * 30, 100, 120 + i * 30, 120] for i in range(len(texts))]
    return [{"index": i, "text": text, "bbox_pixels": box}
            for i, (text, box) in enumerate(zip(texts, boxes))]


def test_physical_inches_equal_fixed_dpi_pixels_and_preserve_pixel_box():
    geometry = adapter.validate_page_geometry(physical_page(), azure_page(),
                                              [0, 0, 2000, 1200], 0)
    assert geometry["render_size"] == [2000, 1200]
    assert geometry["canvas_size"] == [2000, 1200]
    assert geometry["offset_xy"] == [0, 0]
    assert geometry["scale_xy"] == pytest.approx([1, 1])
    assert adapter.map_box([10, 20, 210, 220], geometry) == [10, 20, 210, 220]


def test_uniquely_identified_crop_frame_has_top_origin_translation():
    pdf = physical_page()
    pdf.update(mediabox=[10, 20, 730, 452], cropbox=[82, 56, 658, 416])
    azure = dict(azure_page(), width=8, height=5)
    geometry = adapter.validate_page_geometry(pdf, azure, [0, 0, 1600, 1000], 0)
    # Crop starts 72pt from media left and 36pt below media top: 200px,100px.
    assert geometry["canvas_frame"] == "CropBox"
    assert geometry["render_size"] == [2000, 1200]
    assert geometry["offset_xy"] == pytest.approx([200, 100])
    assert adapter.map_box([10, 20, 210, 220], geometry) == [210, 120, 410, 320]


def test_shifted_equal_pdf_frames_do_not_shift_visible_raster():
    pdf = physical_page()
    pdf.update(mediabox=[10, 20, 730, 452], cropbox=[10, 20, 730, 452])
    geometry = adapter.validate_page_geometry(pdf, azure_page(), [0, 0, 2000, 1200], 0)
    assert geometry["offset_xy"] == [0, 0]
    assert adapter.map_box([50, 60, 70, 80], geometry) == [50, 60, 70, 80]


@pytest.mark.parametrize("field,value", [
    ("rotation", 90), ("rotation", 180), ("rotation", float("nan")),
    ("user_unit", 2), ("user_unit", float("inf")),
    ("mediabox", [0, 0, float("nan"), 432]),
    ("cropbox", [-1, 0, 720, 432]),
    ("cropbox", [0, 0, 721, 432]),
])
def test_unsupported_pdf_frames_rejected_without_fallback(field, value):
    pdf = physical_page()
    pdf[field] = value
    with pytest.raises(adapter.Ineligible):
        adapter.validate_page_geometry(pdf, azure_page(), [0, 0, 2000, 1200], 0)


def test_two_distinct_plausible_pdf_frames_are_ambiguous():
    pdf = physical_page()
    pdf["cropbox"] = [0.1, 0, 720, 432]
    with pytest.raises(adapter.Ineligible, match="ambiguous"):
        adapter.validate_page_geometry(pdf, azure_page(), [0, 0, 2000, 1200], 0)


@pytest.mark.parametrize("field,value", [
    ("unit", "pixel"), ("unit", "point"), ("unit", None),
    ("width", 6), ("height", 10), ("width", True),
    ("height", float("nan")), ("height", float("inf")),
    ("page", 0), ("page", 2),
])
def test_azure_units_size_and_one_based_page_must_agree(field, value):
    raw = azure_page()
    raw[field] = value
    with pytest.raises(adapter.Ineligible):
        adapter.validate_page_geometry(physical_page(), raw, [0, 0, 2000, 1200], 0)


@pytest.mark.parametrize("pdf_field,pdf_value,azure_field,azure_value", [
    ("rotation", False, None, None), ("user_unit", True, None, None),
    ("page_count", True, None, None), (None, None, "page", True),
])
def test_boolean_metadata_is_not_a_numeric_geometry_declaration(
        pdf_field, pdf_value, azure_field, azure_value):
    pdf, raw = physical_page(), azure_page()
    if pdf_field:
        pdf[pdf_field] = pdf_value
    if azure_field:
        raw[azure_field] = azure_value
    with pytest.raises(adapter.Ineligible):
        adapter.validate_page_geometry(pdf, raw, [0, 0, 2000, 1200], 0)


@pytest.mark.parametrize("canvas", [[0, 0, 1200, 2000], [0, 0, 1000, 600],
                                     [10, 0, 2010, 1200], [0, 0, 2000.0, 1200]])
def test_canvas_is_not_a_normalized_or_swapped_coordinate_frame(canvas):
    with pytest.raises(adapter.Ineligible):
        adapter.validate_page_geometry(physical_page(), azure_page(), canvas, 0)


def test_physical_geometry_does_not_bypass_fixed_source_area_floor():
    pdf = dict(physical_page(), mediabox=[0, 0, 720, 360], cropbox=[0, 0, 720, 360])
    raw = dict(azure_page(), height=5)
    with pytest.raises(adapter.Ineligible, match="below_fixed_size"):
        adapter.validate_page_geometry(pdf, raw, [0, 0, 2000, 1000], 0)


@pytest.mark.parametrize("page_index", [-1, 1, True, 0.0])
def test_human_page_index_is_zero_based_and_typed(page_index):
    with pytest.raises(adapter.Ineligible):
        adapter.validate_page_geometry(physical_page(), azure_page(), [0, 0, 2000, 1200], page_index)


def test_original_azure_and_due_tokens_preserve_whole_page_order():
    due, original, geometry = assets()
    selected = adapter.page_tokens(due, original, 0, geometry, DOC_ID, 1)
    assert [x["index"] for x in selected] == [0, 1, 2]
    assert [x["text"] for x in selected] == ["Net", "$5", "million"]
    assert [x["bbox_pixels"] for x in selected] == due["positions"]


@pytest.mark.parametrize("mutation", ["identity", "tokens_length", "page_count",
    "range_gap", "range_negative", "range_past_tokens", "range_incomplete",
    "range_boolean", "word_order", "polygon_nonfinite", "polygon_shape",
    "word_geometry", "box_outside", "box_reversed"])
def test_ocr_corruptions_reject_instead_of_selecting_another_engine(mutation):
    due, original, geometry = assets()
    page = original["analyzeResult"]["readResults"][0]
    word = page["lines"][0]["words"][0]
    if mutation == "identity":
        due["doc_id"] = "another.pdf"
    elif mutation == "tokens_length":
        due["positions"].pop()
    elif mutation == "page_count":
        original["analyzeResult"]["readResults"].append(copy.deepcopy(page))
    elif mutation.startswith("range_"):
        due["structures"]["pages"]["structure_value"] = {
            "range_gap": [[1, 3]], "range_negative": [[-1, 3]],
            "range_past_tokens": [[0, 4]], "range_incomplete": [[0, 2]],
            "range_boolean": [[False, 3]],
        }[mutation]
    elif mutation == "word_order":
        page["lines"][0]["words"].reverse()
    elif mutation == "polygon_nonfinite":
        word["boundingBox"][0] = float("nan")
    elif mutation == "polygon_shape":
        word["boundingBox"].pop()
    elif mutation == "word_geometry":
        word["boundingBox"] = [value + 1 for value in word["boundingBox"]]
    elif mutation == "box_outside":
        due["positions"][0] = [-20, 100, 20, 120]
        word["boundingBox"] = [-.1, .5, .1, .5, .1, .6, -.1, .6]
    else:
        due["positions"][0] = [140, 100, 100, 120]
    with pytest.raises(adapter.Ineligible):
        adapter.page_tokens(due, original, 0, geometry, DOC_ID, 1)


def test_overlapping_or_duplicated_regions_cannot_inflate_coverage():
    word = [100, 100, 120, 120]
    assert adapter.union_coverage(word, [[106, 100, 114, 120]] * 2) == pytest.approx(.4)
    regions = [[106, 100, 114, 120], [107, 100, 115, 120]]
    assert adapter.union_coverage(word, regions) == pytest.approx(.45)
    with pytest.raises(adapter.Ineligible, match="not_supported"):
        adapter.find_supported_span(words(["answer"], [word]), "answer", regions)
    assert adapter.union_coverage(word, [[90, 90, 130, 130]]) == 1
    assert adapter.union_coverage(word, [[200, 200, 210, 210]]) == 0


def test_adjacent_rectangle_intervals_cover_once_without_a_phantom_gap():
    word = [100, 100, 120, 120]
    horizontal_halves = [[100, 100, 110, 120], [110, 100, 120, 120]]
    vertical_halves = [[100, 100, 120, 110], [100, 110, 120, 120]]
    for boxes in (horizontal_halves, vertical_halves):
        assert adapter.union_coverage(word, boxes) == 1
        result = adapter.find_supported_span(words(["answer"], [word]), "answer", boxes)
        assert result["support"][0]["full_coverage"] is True
    separated_quarters = [[100, 100, 110, 110], [110, 110, 120, 120]]
    assert adapter.union_coverage(word, separated_quarters) == .5


def test_declared_half_word_support_is_flagged_as_partial_for_source_review():
    result = adapter.find_supported_span(words(["answer"], [[100, 100, 120, 120]]),
                                         "answer", [[100, 100, 110, 120]])
    assert result["support"][0]["coverage_fraction"] == .5
    assert result["support"][0]["full_coverage"] is False


def test_source_case_and_whitespace_match_without_rewriting_original_text():
    result = adapter.find_supported_span(words(["Net", "$5", "million"]),
                                         " NET\t$5\nMILLION ", [[90, 90, 250, 130]])
    assert result["text"] == "Net $5 million"
    assert [x["text"] for x in result["tokens"]] == ["Net", "$5", "million"]


@pytest.mark.parametrize("visible,reference", [
    ("$5", "5"), ("5%", "5"), ("-5", "5"), ("(5)", "5"),
    ("1,000", "1000"), ("1.5", "15"), ("two-thirds", "thirds"),
    ("1234", "234"), ("1234", "123"), ("two-thirds", "two"),
    ("5", "$5.00"), ("café", "cafe\u0301"),
])
def test_gold_matching_does_not_strip_meaningful_characters_or_partial_tokens(visible, reference):
    with pytest.raises(adapter.Ineligible, match="not_supported"):
        adapter.find_supported_span(words([visible]), reference, [[90, 90, 200, 150]])


def test_region_filter_cannot_splice_nonconsecutive_ocr_into_a_gold_phrase():
    tokens = words(["gross", "not", "profit"],
                   [[100, 100, 120, 120], [800, 800, 820, 820], [140, 100, 170, 120]])
    with pytest.raises(adapter.Ineligible, match="not_supported"):
        adapter.find_supported_span(tokens, "gross profit", [[90, 90, 180, 130]])


def test_repeated_supported_canonical_span_is_not_silently_first_matched():
    with pytest.raises(adapter.Ineligible, match="multiple_supported"):
        adapter.find_supported_span(words(["2020", "2020"]), "2020", [[90, 90, 200, 150]])
    with pytest.raises(adapter.Ineligible, match="not_supported"):
        adapter.find_supported_span(words(["2020"]), "2020", [[500, 500, 600, 600]])


def test_human_roi_preserves_context_instead_of_shrinking_to_answer_words():
    roi, audit = adapter.build_human_roi([[400, 400, 650, 650]],
                                         [[500, 500, 520, 520]], [2000, 1200])
    assert roi == [360, 360, 690, 690]
    assert audit["margin_pixels_each_side"] == 40
    assert audit["page_area_fraction"] == pytest.approx(108900 / 2400000)


def test_minimum_roi_shifts_inside_page_at_bottom_right():
    roi, _ = adapter.build_human_roi([[1940, 1140, 1980, 1180]],
                                     [[1950, 1150, 1960, 1160]], [2000, 1200])
    assert roi == [1744, 944, 2000, 1200]


def test_large_evidence_union_is_not_resized_to_meet_area_quota():
    with pytest.raises(adapter.Ineligible, match="quarter_page"):
        adapter.build_human_roi([[200, 100, 1800, 1100]],
                                [[500, 500, 520, 520]], [2000, 1200])


def test_final_padded_region_must_enclose_every_supported_word():
    with pytest.raises(adapter.Ineligible, match="enclose_evidence"):
        adapter.build_human_roi([[400, 400, 420, 420]],
                                [[900, 400, 920, 420]], [2000, 1200])


@pytest.mark.parametrize("field,value", [("left", -1), ("width", 0),
    ("height", float("nan")), ("page", True), ("page", -1), ("page", 1)])
def test_raw_human_boxes_fail_closed_through_complete_assessment(field, value):
    q = question()
    q["answers_page_bounding_boxes"][0][0][field] = value
    with pytest.raises(adapter.Ineligible):
        assess(q)


def test_supported_variant_never_rescues_wrong_canonical_reference():
    q = question("$5.00 million")
    q["answers_variants"] = ["$5 million"]
    with pytest.raises(adapter.Ineligible, match="not_supported"):
        assess(q)


def test_assessment_keeps_unsupported_variants_and_semantic_review_pending():
    q = question()
    q["answers_variants"] = ["$5 million", "$5.00 million", "5 million"]
    result = assess(q)
    assert q["answers"] == ["$5 million"]
    assert result["original_answer_variants"] == q["answers_variants"]
    assert result["ocr_supported_answers"] == ["$5 million"]
    assert [x["supported"] for x in result["variant_checks"]] == [True, False, False]
    assert result["validated_primary_answers"] == []
    assert result["source_semantic_audit_status"] == "pending"
