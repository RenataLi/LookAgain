"""Source-only DUDE adapter: technical census, audit renders and reviewed cohort.

Never loads a model or experiment outcomes. Canonical source answers remain
unchanged; OCR support is technical evidence, not semantic approval.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
from numbers import Real
from pathlib import Path
import re
import statistics
import subprocess
import time
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
SEED = 20260927
PROBE_DOC_ID = "0017b64bd017f06db47e56a6a113e22e"
GEOMETRY_TOLERANCE_PIXELS = 2.0
PINS = {
    "annotations": "4f883956ba02cb676022ec24878cdc415e759438e945c2d479bdae488873e2e8",
    "lineage": "fbe2888343b3b7f137201b679355a54035b70cebddba19ed0d0374807b2299e9",
    "metadata_helper": "b45705088884a235623a583d6f1b97996394a660918fb365fccda0c1eaae3527",
    "prior_main": "757c3a4d024ac726f701c8732424e3457bfb292d224f67e2de22201ac641adc5",
    "prior_smoke": "ac4f35d9edf4c6e6a7b70225668fc89172a618f3eaecaa661009269c2c109564",
    "tatdqa_json": "3025b4ca9c3c87f5ecd2c504371e1ee92929ef1f31d8a3ac38bc3a5376582fab",
    "tatdqa_zip": "412dde804ec9d5888b888a9d7179f2157bc2679b92379f9804e6f2062e7237d9",
}


class Ineligible(ValueError):
    """A logged source-only exclusion, not a silent fallback."""


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_pin(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Frozen input hash differs: {path.name}")
    return actual


def rank(value: str) -> tuple[str, str]:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest(), value


def valid_box(box) -> bool:
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(x) in (int, float) and math.isfinite(x) for x in box)
            and box[0] < box[2] and box[1] < box[3])


def contains(outer, inner) -> bool:
    return (valid_box(outer) and valid_box(inner)
            and outer[0] <= inner[0] < inner[2] <= outer[2]
            and outer[1] <= inner[1] < inner[3] <= outer[3])


def validate_page_geometry(pdf: dict, azure_page: dict, due_canvas: list,
                           page_index: int) -> dict:
    """Reconcile Azure physical inches/DUE canvas with PDF CropBox or MediaBox."""
    media, crop = pdf["mediabox"], pdf["cropbox"]
    if type(pdf["page_count"]) is not int or pdf["page_count"] < 1:
        raise Ineligible("invalid_pdf_page_count")
    if (not valid_box(media) or not valid_box(crop) or not contains(media, crop)):
        raise Ineligible("unsupported_pdf_boxes")
    if type(pdf["rotation"]) is not int or pdf["rotation"] != 0:
        raise Ineligible("unsupported_pdf_rotation")
    if type(pdf["user_unit"]) not in (int, float) or pdf["user_unit"] != 1:
        raise Ineligible("unsupported_pdf_user_unit")
    if type(page_index) is not int or not 0 <= page_index < pdf["page_count"]:
        raise Ineligible("annotated_page_out_of_range")
    if type(azure_page.get("page")) is not int or azure_page.get("page") != page_index + 1:
        raise Ineligible("azure_page_number_mismatch")
    if azure_page.get("unit") != "inch":
        raise Ineligible("azure_unit_not_inch")
    physical = [azure_page.get("width"), azure_page.get("height")]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in physical):
        raise Ineligible("invalid_azure_physical_size")
    if not valid_box(due_canvas) or due_canvas[:2] != [0, 0]:
        raise Ineligible("unsupported_due_canvas")
    canvas = [due_canvas[2], due_canvas[3]]
    if any(type(v) is not int for v in canvas):
        raise Ineligible("noninteger_due_canvas")
    if any(abs(physical[i] * 200 - canvas[i]) > GEOMETRY_TOLERANCE_PIXELS for i in (0, 1)):
        raise Ineligible("azure_due_physical_size_mismatch")
    frames = []
    for label, frame in (("MediaBox", media), ("CropBox", crop)):
        dims = [frame[2] - frame[0], frame[3] - frame[1]]
        if all(abs(dims[i] * 200 / 72 - canvas[i]) <= GEOMETRY_TOLERANCE_PIXELS for i in (0, 1)):
            frames.append((label, frame))
    if not frames:
        raise Ineligible("azure_pdf_canvas_mismatch")
    if len(frames) == 2 and media != crop:
        raise Ineligible("ambiguous_pdf_canvas_frame")
    label, frame = frames[0]
    render = [math.ceil((media[2] - media[0]) * 200 / 72),
              math.ceil((media[3] - media[1]) * 200 / 72)]
    if min(render) < 256 or math.prod(render) < 2 * 1024 * 32 * 32:
        raise Ineligible("source_render_below_fixed_size")
    mx, my = render[0] / (media[2] - media[0]), render[1] / (media[3] - media[1])
    return {
        "render_size": render, "canvas_size": canvas, "canvas_frame": label,
        "scale_xy": [(frame[2] - frame[0]) / (media[2] - media[0]) * (render[0] / canvas[0]),
                     (frame[3] - frame[1]) / (media[3] - media[1]) * (render[1] / canvas[1])],
        "offset_xy": [(frame[0] - media[0]) * mx, (media[3] - frame[3]) * my],
        "physical_unit": "inch", "render_dpi": 200,
        "page_index_convention": "Raw human page index is zero-based; Azure page field is index+1; source support and visual audit are required.",
        "max_geometry_tolerance_pixels": GEOMETRY_TOLERANCE_PIXELS,
    }


def map_box(box: list, geometry: dict) -> list[int]:
    canvas = [0, 0, *geometry["canvas_size"]]
    if not contains(canvas, box):
        raise Ineligible("box_outside_azure_canvas")
    sx, sy = geometry["scale_xy"]
    dx, dy = geometry["offset_xy"]
    coordinates = [dx + box[0] * sx, dy + box[1] * sy,
                   dx + box[2] * sx, dy + box[3] * sy]
    # Correct floating arithmetic noise only; this is not a geometry tolerance.
    coordinates = [round(value) if abs(value - round(value)) <= 1e-9 else value
                   for value in coordinates]
    result = [math.floor(coordinates[0]), math.floor(coordinates[1]),
              math.ceil(coordinates[2]), math.ceil(coordinates[3])]
    if not contains([0, 0, *geometry["render_size"]], result):
        raise Ineligible("mapped_box_outside_source")
    return result


def page_tokens(due: dict, original: dict, page_index: int,
                geometry: dict, doc_id: str, pdf_page_count: int) -> list[dict]:
    if due.get("doc_id") != doc_id + ".pdf":
        raise Ineligible("ocr_document_identity_mismatch")
    tokens, positions = due.get("tokens"), due.get("positions")
    structure = due.get("structures", {}).get("pages", {})
    ranges, canvases = structure.get("structure_value"), structure.get("positions")
    originals = original.get("analyzeResult", {}).get("readResults")
    if (not isinstance(tokens, list) or not isinstance(positions, list) or len(tokens) != len(positions)
            or not isinstance(ranges, list) or not isinstance(canvases, list)
            or not isinstance(originals, list)
            or len(ranges) != pdf_page_count or len(canvases) != pdf_page_count
            or len(originals) != pdf_page_count):
        raise Ineligible("ocr_pdf_page_or_token_count_mismatch")
    cursor = 0
    for start_end in ranges:
        if (not isinstance(start_end, list) or len(start_end) != 2
                or any(type(x) is not int for x in start_end)
                or start_end[0] != cursor or not start_end[0] <= start_end[1] <= len(tokens)):
            raise Ineligible("invalid_due_page_token_ranges")
        cursor = start_end[1]
    if cursor != len(tokens):
        raise Ineligible("incomplete_due_page_token_ranges")
    start, end = ranges[page_index]
    words = [word for line in originals[page_index].get("lines", []) for word in line.get("words", [])]
    if [word.get("text") for word in words] != tokens[start:end]:
        raise Ineligible("azure_original_due_word_alignment_failure")
    result = []
    for local_index, word in enumerate(words):
        token_index = start + local_index
        bbox = positions[token_index]
        polygon = word.get("boundingBox")
        if (not isinstance(polygon, list) or len(polygon) != 8
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in polygon)):
            raise Ineligible("invalid_azure_original_word_polygon")
        physical_bbox = [min(polygon[0::2]) * 200, min(polygon[1::2]) * 200,
                         max(polygon[0::2]) * 200, max(polygon[1::2]) * 200]
        if not valid_box(bbox) or any(abs(bbox[i] - physical_bbox[i]) > GEOMETRY_TOLERANCE_PIXELS for i in range(4)):
            raise Ineligible("azure_original_due_word_geometry_mismatch")
        if not isinstance(tokens[token_index], str) or not normalize(tokens[token_index]):
            raise Ineligible("invalid_ocr_word_text")
        result.append({"index": token_index, "text": tokens[token_index],
                       "bbox_due": bbox, "bbox_pixels": map_box(bbox, geometry)})
    return result


def union_coverage(word_box: list, boxes: list[list]) -> float:
    """Exact rectangle-union intersection area, without double-counting overlap."""
    if not valid_box(word_box) or not all(valid_box(box) for box in boxes):
        raise Ineligible("invalid_overlap_geometry")
    clips = [[max(word_box[0], b[0]), max(word_box[1], b[1]),
              min(word_box[2], b[2]), min(word_box[3], b[3])] for b in boxes]
    clips = [box for box in clips if valid_box(box)]
    xs = sorted({box[i] for box in clips for i in (0, 2)})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((box[1], box[3]) for box in clips if box[0] < right and box[2] > left)
        covered, lower, upper = 0.0, None, None
        for y0, y1 in intervals:
            if lower is None:
                lower, upper = y0, y1
            elif y0 > upper:
                covered += upper - lower
                lower, upper = y0, y1
            else:
                upper = max(upper, y1)
        if lower is not None:
            covered += upper - lower
        total += (right - left) * covered
    return total / ((word_box[2] - word_box[0]) * (word_box[3] - word_box[1]))


def find_supported_span(tokens: list[dict], answer: str, human_boxes: list[list]) -> dict:
    target = normalize(answer)
    if not target:
        raise Ineligible("empty_reference")
    matches = []
    for start in range(len(tokens)):
        text = ""
        for end in range(start, len(tokens)):
            text = (text + " " + normalize(tokens[end]["text"])).strip()
            if not target.startswith(text):
                break
            if text == target:
                selected, support = tokens[start:end + 1], []
                for token in selected:
                    box = token["bbox_pixels"]
                    center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                    inside = any(b[0] <= center[0] <= b[2] and b[1] <= center[1] <= b[3] for b in human_boxes)
                    coverage = union_coverage(box, human_boxes)
                    support.append({"token_index": token["index"], "center_inside": inside,
                                    "coverage_fraction": coverage, "full_coverage": coverage >= 1 - 1e-12})
                if all(s["center_inside"] and s["coverage_fraction"] >= .5 for s in support):
                    matches.append({"tokens": selected, "support": support,
                                    "text": " ".join(token["text"] for token in selected)})
                break
    if not matches:
        raise Ineligible("canonical_reference_not_supported_in_human_region")
    if len(matches) != 1:
        raise Ineligible("multiple_supported_reference_occurrences")
    return matches[0]


def build_human_roi(human_boxes: list[list], matched_word_boxes: list[list],
                    render_size: list[int]) -> tuple[list[int], dict]:
    page = [0, 0, *render_size]
    if not human_boxes or not matched_word_boxes or not all(contains(page, box) for box in human_boxes + matched_word_boxes):
        raise Ineligible("evidence_geometry_outside_source")
    raw = [min(b[0] for b in human_boxes), min(b[1] for b in human_boxes),
           max(b[2] for b in human_boxes), max(b[3] for b in human_boxes)]
    median_height = statistics.median(box[3] - box[1] for box in matched_word_boxes)
    margin = 2 * median_height
    width, height = render_size
    if min(render_size) < 256:
        raise Ineligible("source_below_minimum_roi")
    roi_width = min(width, max(256, math.ceil(raw[2] - raw[0] + 2 * margin)))
    roi_height = min(height, max(256, math.ceil(raw[3] - raw[1] + 2 * margin)))
    left = min(width - roi_width, max(0, math.floor((raw[0] + raw[2] - roi_width) / 2)))
    top = min(height - roi_height, max(0, math.floor((raw[1] + raw[3] - roi_height) / 2)))
    roi = [left, top, left + roi_width, top + roi_height]
    if not all(contains(roi, box) for box in human_boxes + matched_word_boxes):
        raise Ineligible("padded_human_roi_does_not_enclose_evidence")
    fraction = roi_width * roi_height / (width * height)
    if fraction > .25:
        raise Ineligible("human_roi_exceeds_quarter_page")
    return roi, {"basis": "Union of ALL original human evidence rectangles",
                 "human_union_pixels": raw, "median_matched_gold_word_height": median_height,
                 "margin_pixels_each_side": margin, "minimum_side_pixels": 256,
                 "page_area_fraction": fraction, "no_shrink_to_quota": True}


def assess_question(question: dict, doc_id: str, pdf_pages: list[dict],
                    due: dict, original: dict, metadata_helper) -> dict:
    if not isinstance(question.get("question"), str) or not question["question"].strip():
        raise Ineligible("empty_or_nonstring_question")
    failure = metadata_helper.short_extractive_failure(question)
    if failure:
        raise Ineligible(failure)
    boxes_info = metadata_helper.inspect_boxes(question.get("answers_page_bounding_boxes"))
    if not boxes_info["valid"]:
        raise Ineligible(boxes_info["reason"])
    if not boxes_info["same_annotated_page"]:
        raise Ineligible("multiple_annotated_answer_pages")
    page_index = boxes_info["pages"][0]
    if not 0 <= page_index < len(pdf_pages):
        raise Ineligible("annotated_page_out_of_range")
    originals = original.get("analyzeResult", {}).get("readResults", [])
    canvases = due.get("structures", {}).get("pages", {}).get("positions", [])
    if len(originals) != len(pdf_pages) or len(canvases) != len(pdf_pages):
        raise Ineligible("ocr_pdf_page_count_mismatch")
    geometry = validate_page_geometry(pdf_pages[page_index], originals[page_index],
                                      canvases[page_index], page_index)
    tokens = page_tokens(due, original, page_index, geometry, doc_id, len(pdf_pages))
    raw_boxes = question["answers_page_bounding_boxes"][0]
    human_boxes = [map_box([b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"]], geometry)
                   for b in raw_boxes]
    canonical = question["answers"][0]
    match = find_supported_span(tokens, canonical, human_boxes)
    roi, construction = build_human_roi(human_boxes,
                                        [word["bbox_pixels"] for word in match["tokens"]],
                                        geometry["render_size"])
    variants = question.get("answers_variants", [])
    if not isinstance(variants, list) or any(not isinstance(x, str) for x in variants):
        raise Ineligible("invalid_original_answer_variants")
    supported, variant_checks = [canonical], []
    for variant in variants:
        try:
            variant_match = find_supported_span(tokens, variant, human_boxes)
            if not all(contains(roi, word["bbox_pixels"]) for word in variant_match["tokens"]):
                raise Ineligible("variant_support_outside_fixed_roi")
            if variant not in supported:
                supported.append(variant)
            variant_checks.append({"answer": variant, "supported": True, "match": variant_match})
        except Ineligible as error:
            variant_checks.append({"answer": variant, "supported": False, "reason": str(error)})
    return {
        "question_uid": question["questionId"], "doc_id": doc_id, "question": question["question"],
        "answer": canonical, "original_answers": list(question["answers"]),
        "original_answer_variants": variants, "ocr_supported_answers": supported,
        "validated_primary_answers": [], "answer_page_index": page_index,
        "answer_page_privileged": True, "roi_privileged": True, "roi_pixels": roi,
        "pixel_dimensions": geometry["render_size"], "pdf_page_count": len(pdf_pages),
        "geometry": geometry, "human_boxes_original": raw_boxes,
        "human_boxes_pixels": human_boxes, "canonical_match": match,
        "variant_checks": variant_checks, "roi_construction": construction,
        "source_semantic_audit_status": "pending",
    }


def _metadata_helper():
    path = PROJECT / "experiments/profile_dude_metadata.py"
    check_pin(path, PINS["metadata_helper"])
    spec = importlib.util.spec_from_file_location("dude_preparation_metadata_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _asset_index(assets: Path) -> tuple[dict, dict]:
    metadata_path, manifest_path = assets / "extraction.json", assets / "extracted_assets.jsonl"
    extraction = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (extraction.get("archive_sha256") != "1506384a93022a2da6b180270345a6928ea7347842eaa2d2e177190c4fd29cae"
            or extraction.get("author_revision") != "b3662175d3b2482d711f18559b7acc2a5bccc600"
            or extraction.get("verified_complete") is not True
            or extraction.get("archive_fully_scanned") is not True):
        raise ValueError("A verified, fully scanned pinned source archive is required")
    if sha256_file(manifest_path) != extraction["asset_manifest_sha256"]:
        raise ValueError("Extracted-asset journal hash differs")
    index = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = row["doc_id"], row["kind"]
        if key in index:
            raise ValueError("Duplicate extracted asset")
        relative = Path(row["relative_path"])
        target = (assets / relative).resolve()
        if relative.is_absolute() or relative.drive or not target.is_relative_to(assets):
            raise ValueError("Unsafe extracted asset path")
        if target.is_symlink():
            raise ValueError("Symlinked source asset")
        index[key] = {**row, "path": target}
    return index, {
        "archive_sha256": extraction["archive_sha256"],
        "author_revision": extraction["author_revision"],
        "verified_complete": True, "archive_fully_scanned": True,
        "extraction_metadata_sha256": sha256_file(metadata_path),
        "asset_manifest_sha256": extraction["asset_manifest_sha256"],
    }


def _read_bound_asset(item: dict) -> bytes:
    blob = item["path"].read_bytes()
    if len(blob) != item["bytes"] or hashlib.sha256(blob).hexdigest() != item["sha256"]:
        raise ValueError("Extracted source asset differs from its verified journal")
    return blob


def _pdf_number(value, *, integer: bool = False):
    """Normalize PDF-library numeric scalars without coercing invalid metadata.

    pypdf NumberObject/FloatObject subclass int/float. BooleanObject and PDF
    strings are not numbers; fractional rotation is never truncated to zero.
    None deliberately reaches the unchanged strict geometry validator.
    """
    if hasattr(value, "get_object"):
        value = value.get_object()
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return None
    if integer:
        return int(value) if value == int(value) else None
    return float(value)


def _pdf_box(value):
    if hasattr(value, "get_object"):
        value = value.get_object()
    return [_pdf_number(v) for v in value] if isinstance(value, (list, tuple)) else None


def _pdf_metadata(blob: bytes):
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(blob))
    page_count = len(reader.pages)
    pages, texts, text_failures = [], [], []
    for index, page in enumerate(reader.pages):
        # Read raw scalars before RectangleObject or float() could coerce them.
        # PdfReader has already resolved inherited page-tree geometry here.
        media = page.get("/MediaBox")
        pages.append({
            "page_count": page_count, "mediabox": _pdf_box(media),
            "cropbox": _pdf_box(page.get("/CropBox", media)),
            "rotation": _pdf_number(page.get("/Rotate", 0), integer=True),
            "user_unit": _pdf_number(page.get("/UserUnit", 1)),
        })
        try:
            texts.append(normalize(page.extract_text() or ""))
        except Exception as error:
            texts.append("")
            text_failures.append({"page": index, "error_type": type(error).__name__})
    meaningful = sum(len(text) for text in texts)
    text_tokens = " ".join(texts).split()
    sketch = sorted({hashlib.blake2b(" ".join(text_tokens[i:i + 5]).encode("utf-8"), digest_size=8).hexdigest()
                     for i in range(max(0, len(text_tokens) - 4))})[:64] if not text_failures else []
    return pages, {
        "full_pdf_text_sha256": canonical_hash(texts) if meaningful >= 200 and not text_failures else None,
        "normalized_text_chars": meaningful, "text_extraction_failures": text_failures,
        "bottom64_five_word_shingle_sketch": sketch,
        "text_duplicate_rule": "Casefold/whitespace per full PDF page; preserve ordered page boundaries, every digit/punctuation/sign; key only when total>=200 chars and all pages extracted.",
    }


def _historical_sources(tatdqa_raw: Path | None):
    rows = []
    bindings = {}
    for label, filename in (("prior_main", "main_manifest.jsonl"), ("prior_smoke", "smoke_manifest.jsonl")):
        path = PROJECT / "reports/native100/provenance" / filename
        bindings[label] = check_pin(path, PINS[label])
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    names = {row["source_id"] for row in rows}
    hashes = {row["source_pdf_sha256"] for row in rows}
    coverage = "102 previously selected source PDFs"
    if tatdqa_raw is not None:
        annotation, archive_path = tatdqa_raw / "tat_dqa_dataset_train.json", tatdqa_raw / "tat_dqa_train.zip"
        bindings["tatdqa_json"] = check_pin(annotation, PINS["tatdqa_json"])
        bindings["tatdqa_zip"] = check_pin(archive_path, PINS["tatdqa_zip"])
        entries = json.loads(annotation.read_text(encoding="utf-8"))
        with zipfile.ZipFile(archive_path) as archive:
            for entry in entries:
                if entry["doc"]["source"] in names:
                    uid = entry["doc"]["uid"]
                    if not re.fullmatch("[a-f0-9]{32}", uid):
                        raise ValueError("Unsafe historical document UID")
                    hashes.add(hashlib.sha256(archive.read(f"train/{uid}.pdf")).hexdigest())
        coverage = "Every released PDF fragment of all102 historical original TAT-DQA reports"
    return names, hashes, {"bindings": bindings, "report_sources": sorted(names),
                           "known_pdf_sha256": sorted(hashes), "hash_coverage": coverage}


def source_resource_keys(url: str, helper) -> list[str]:
    from urllib.parse import unquote, urlsplit
    normalized = helper.normalized_url(url)
    if not normalized:
        return []
    keys = ["url:" + normalized]
    parsed = urlsplit(url)
    host, path = (parsed.hostname or "").casefold(), unquote(parsed.path)
    if host.endswith("documentcloud.org"):
        match = re.search(r"/documents/(\d+)", path)
        if match:
            keys.append("documentcloud:" + match[1])
    if host.endswith("archive.org"):
        match = re.search(r"/(?:details|download)/([^/]+)", path)
        if match:
            keys.append("archive-item:" + match[1])
    if host == "commons.wikimedia.org" and "/wiki/File:" in path:
        keys.append("wikimedia-file:" + path.split("/wiki/File:", 1)[1].replace("_", " "))
    return keys


class DisjointSources:
    def __init__(self, ids):
        self.parent = {value: value for value in ids}
        self.edges = []

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def join(self, left, right, reason):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)
            self.edges.append({"left": left, "right": right, "reason": reason})

    def clusters(self):
        groups = defaultdict(list)
        for doc_id in self.parent:
            groups[self.find(doc_id)].append(doc_id)
        return [sorted(group) for group in groups.values()]


def near_duplicate_flags(documents: dict, clusters: DisjointSources) -> list[dict]:
    """Review flags only; no merge or eligibility decision from text similarity."""
    sketches = [(doc_id, set(info.get("bottom64_five_word_shingle_sketch", [])))
                for doc_id, info in documents.items()
                if len(info.get("bottom64_five_word_shingle_sketch", [])) >= 48]
    result = []
    for index, (left, a) in enumerate(sketches):
        for right, b in sketches[index + 1:]:
            if clusters.find(left) == clusters.find(right):
                continue
            shared = len(a & b)
            if shared >= 48:
                result.append({"left_doc_id": left, "right_doc_id": right,
                               "shared_bottom64_shingles": shared,
                               "automatic_merge": False, "source_review_required": True})
    return result


def assemble_cluster_candidates(docs: dict, matches: dict, clusters: DisjointSources, asset_index: dict) -> dict:
    near_flags = near_duplicate_flags(docs, clusters)
    near_by_doc = defaultdict(list)
    for flag in near_flags:
        near_by_doc[flag["left_doc_id"]].append(flag)
        near_by_doc[flag["right_doc_id"]].append(flag)
    cluster_rows, candidates = [], []
    for members in clusters.clusters():
        cluster_id = "dude-source:" + canonical_hash(members)
        flags = []
        if any(docs[doc]["prior_source_matches"] for doc in members):
            flags.append("prior_tatdqa_source_overlap")
        is_probe = PROBE_DOC_ID in members
        available = [doc for doc in members if doc in matches and (docs[doc]["lineage_verified"] or is_probe)]
        if not available:
            flags.append("no_eligible_question_with_verified_lineage")
        cluster_row = {"source_cluster_id": cluster_id, "doc_ids": members,
                       "prior_engineering_cluster": is_probe, "exclusion_reasons": flags}
        cluster_rows.append(cluster_row)
        if not flags:
            doc_id = min(available, key=lambda doc: (rank(matches[doc]["question_uid"]), rank(doc)))
            candidate = matches[doc_id]
            candidate.update(
                example_id="dude:" + candidate["question_uid"], source_id=cluster_id,
                source_cluster_id=cluster_id, source_cluster_doc_ids=members,
                source_urls=docs[doc_id]["source_urls"],
                source_pdf_sha256=docs[doc_id]["source_pdf_sha256"],
                source_ocr_sha256=docs[doc_id]["source_ocr_sha256"],
                source_azure_original_sha256=docs[doc_id]["source_azure_original_sha256"],
                original_pdf_relative_path=asset_index[(doc_id, "pdf")]["relative_path"],
                source_split="train", split="replication", prior_engineering_cluster=is_probe,
                lineage_verified=docs[doc_id]["lineage_verified"],
                near_duplicate_review_flags=[flag for member in members for flag in near_by_doc[member]],
            )
            candidates.append(candidate)
    candidates.sort(key=lambda row: rank(row["source_cluster_id"]))
    new_candidates = [row for row in candidates if not row["prior_engineering_cluster"]]
    reserved_ids = {row["source_cluster_id"] for row in new_candidates[:10]}
    reserved_ids.update(row["source_cluster_id"] for row in cluster_rows if row["prior_engineering_cluster"])
    for index, row in enumerate(candidates):
        row["source_rank"] = index + 1
        row["cohort_reservation"] = ("engineering" if row["prior_engineering_cluster"] or row["source_cluster_id"] in reserved_ids else "main_candidate")
        row["technical_provenance_sha256"] = canonical_hash({key: value for key, value in row.items() if key != "technical_provenance_sha256"})
    return {"clusters": cluster_rows, "candidates": candidates, "reserved_ids": reserved_ids, "near_flags": near_flags, "new_candidates": new_candidates}


def technical_census(metadata: Path, lineage: Path, assets: Path, output: Path,
                     tatdqa_raw: Path | None = None) -> dict:
    started = time.perf_counter()
    metadata, lineage, assets, output = [Path(p).resolve() for p in (metadata, lineage, assets, output)]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Census output must be absent or empty")
    bindings = {"annotations": check_pin(metadata, PINS["annotations"]),
                "lineage": check_pin(lineage, PINS["lineage"]),
                "adapter_sha256": sha256_file(Path(__file__))}
    helper = _metadata_helper()
    asset_index, extraction = _asset_index(assets)
    prior_names, prior_pdf_hashes, historical = _historical_sources(tatdqa_raw)
    data = json.loads(metadata.read_text(encoding="utf-8"))["data"]
    training = [row for row in data if row.get("data_split") == "train"]
    questions_by_doc, metadata_walk = defaultdict(list), []
    for row in training:
        failure = helper.short_extractive_failure(row)
        if failure is None:
            boxes = helper.inspect_boxes(row.get("answers_page_bounding_boxes"))
            if not boxes["valid"]:
                failure = boxes["reason"]
            elif not boxes["same_annotated_page"]:
                failure = "multiple_annotated_answer_pages"
        metadata_walk.append({"question_uid": row["questionId"], "doc_id": row["docId"],
                              "eligible": failure is None, "first_failure": failure})
        if failure is None:
            questions_by_doc[row["docId"]].append(row)
    with lineage.open(encoding="utf-8-sig", newline="") as stream:
        lineage_rows = list(csv.DictReader(stream))
    urls_by_doc = defaultdict(list)
    for row in lineage_rows:
        doc_id, url = row["url_md5 (filename)"].strip(), row["url"].strip()
        if doc_id in questions_by_doc and helper.normalized_url(url):
            urls_by_doc[doc_id].append(url)
    ids = sorted(questions_by_doc)
    clusters = DisjointSources(ids)
    key_owner, docs, question_walk, matches = {}, {}, [], {}
    from urllib.parse import unquote, urlsplit
    old_labels = {helper.compact_label(source[:-4]): source for source in prior_names}
    for doc_index, doc_id in enumerate(ids):
        info = {
            "doc_id": doc_id, "source_urls": sorted(set(urls_by_doc.get(doc_id, []))),
            "lineage_verified": bool(urls_by_doc.get(doc_id)), "technical_failure": None,
            "prior_source_matches": [], "candidate_count": 0, "engineering_probe": doc_id == PROBE_DOC_ID,
        }
        docs[doc_id] = info
        for url in info["source_urls"]:
            label = helper.compact_label(re.sub(r"\.pdf$", "", unquote(urlsplit(url).path.rsplit("/", 1)[-1]), flags=re.I))
            if label in old_labels:
                info["prior_source_matches"].append(old_labels[label])
            for key in source_resource_keys(url, helper):
                if key in key_owner:
                    clusters.join(doc_id, key_owner[key], key)
                else:
                    key_owner[key] = doc_id
        try:
            missing = [kind for kind in ("pdf", "azure_due", "azure_original") if (doc_id, kind) not in asset_index]
            if missing:
                raise Ineligible("missing_verified_assets:" + ",".join(missing))
            raw_assets = {kind: _read_bound_asset(asset_index[(doc_id, kind)])
                          for kind in ("pdf", "azure_due", "azure_original")}
            info["source_pdf_sha256"] = hashlib.sha256(raw_assets["pdf"]).hexdigest()
            info["source_ocr_sha256"] = hashlib.sha256(raw_assets["azure_due"]).hexdigest()
            info["source_azure_original_sha256"] = hashlib.sha256(raw_assets["azure_original"]).hexdigest()
            if info["source_pdf_sha256"] in prior_pdf_hashes:
                info["prior_source_matches"].append("exact_historical_tatdqa_pdf_sha256")
            try:
                pdf_pages, text_metadata = _pdf_metadata(raw_assets["pdf"])
                due, original = json.loads(raw_assets["azure_due"]), json.loads(raw_assets["azure_original"])
            except Exception as error:
                raise Ineligible("source_parse_failure:" + type(error).__name__) from error
            info.update(text_metadata)
            info["pdf_page_count"] = len(pdf_pages)
            for key in ("pdf:" + info["source_pdf_sha256"],
                        "pdf-text:" + info["full_pdf_text_sha256"] if info["full_pdf_text_sha256"] else None):
                if key is not None:
                    if key in key_owner:
                        clusters.join(doc_id, key_owner[key], key)
                    else:
                        key_owner[key] = doc_id
            # Keep the complete source-only QA walk; semantic rejection later
            # rejects the whole source, never triggers question fishing.
            doc_matches = []
            for question in sorted(questions_by_doc[doc_id], key=lambda q: rank(q["questionId"])):
                decision = {"doc_id": doc_id, "question_uid": question["questionId"],
                            "technical_eligible": False, "first_failure": None}
                try:
                    candidate = assess_question(question, doc_id, pdf_pages, due, original, helper)
                    candidate["raw_question_annotation"] = question
                    candidate["pdf_page_geometry"] = pdf_pages[candidate["answer_page_index"]]
                    doc_matches.append(candidate)
                    decision["technical_eligible"] = True
                except (Ineligible, KeyError, TypeError, IndexError, AttributeError, ValueError) as error:
                    decision["first_failure"] = str(error) if isinstance(error, Ineligible) else "malformed_ocr_schema:" + type(error).__name__
                question_walk.append(decision)
            info["candidate_count"] = len(doc_matches)
            if doc_matches:
                # Canonical first technical question per original document.
                matches[doc_id] = doc_matches[0]
            else:
                info["technical_failure"] = "no_technically_eligible_question"
        except Ineligible as error:
            info["technical_failure"] = str(error)
            for question in sorted(questions_by_doc[doc_id], key=lambda q: rank(q["questionId"])):
                question_walk.append({"doc_id": doc_id, "question_uid": question["questionId"],
                                      "technical_eligible": False, "first_failure": "document_failure:" + str(error)})
        if (doc_index + 1) % 100 == 0:
            print(f"Source-only census: {doc_index + 1}/{len(ids)} documents", flush=True)
    assembled = assemble_cluster_candidates(docs, matches, clusters, asset_index)
    cluster_rows, candidates = assembled["clusters"], assembled["candidates"]
    reserved_ids, near_flags = assembled["reserved_ids"], assembled["near_flags"]
    new_candidates = assembled["new_candidates"]
    result = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "source_only_technical_census", "seed": SEED,
        "bindings": bindings, "extraction": extraction, "historical_exclusion": historical,
        "counts": {
            "training_rows": len(training), "metadata_candidate_questions": sum(len(v) for v in questions_by_doc.values()),
            "metadata_candidate_documents": len(ids), "source_clusters": len(cluster_rows),
            "technical_candidates_one_per_cluster": len(candidates),
            "new_candidates_with_verified_lineage": len(new_candidates),
            "engineering_reserved": sum(row["cohort_reservation"] == "engineering" for row in candidates),
            "main_candidate_sources_before_semantic_review": sum(row["cohort_reservation"] == "main_candidate" for row in candidates),
            "document_failure_counts": dict(Counter(row["technical_failure"] for row in docs.values() if row["technical_failure"])),
            "question_first_failure_counts": dict(Counter(row["first_failure"] for row in question_walk if row["first_failure"])),
            "selected_candidate_canvas_frames": dict(Counter(row["geometry"]["canvas_frame"] for row in candidates)),
            "selected_candidates_with_partial_word_support": sum(any(not word["full_coverage"] for word in row["canonical_match"]["support"]) for row in candidates),
            "near_duplicate_review_pairs": len(near_flags),
        },
        "rules": {
            "canonical_reference_mandatory": True, "semantic_reject_excludes_entire_cluster": True,
            "first_technical_question_per_document": True,
            "cluster_document_choice": "Minimum SHA256(seed:questionId) among first technical-pass questions of cluster documents with known lineage; document hash breaks ties; never compare model answers.",
            "main_target": 660, "additional_engineering_reservations": 10,
            "prior_engineering_probe": PROBE_DOC_ID,
            "roi_basis": "ALL human rectangles union plus2 median matched canonical word heights; minimum256square, maximum25% source area; no shrink.",
            "review_required_for_every_final_source": True,
            "duplicate_text_rule": "Full ordered normalized PDF-page text, minimum200 characters, no extraction errors.",
            "partial_word_support": "Every center inside human region and >=50% true union area; partial boxes flagged; fixed padded ROI must fully enclose all matched words.",
        },
        "documents": [docs[doc] for doc in ids], "clusters": sorted(cluster_rows, key=lambda row: rank(row["source_cluster_id"])),
        "source_identity_edges": clusters.edges, "candidates": candidates,
        "engineering_reserved_source_clusters": sorted(reserved_ids),
        "near_duplicate_review_flags": near_flags,
        "near_duplicate_rule": "Bottom64 unique BLAKE2b64 hashes of contiguous5-word normalized full-PDF text shingles; >=48 shared hashes flags source review across distinct exact-source clusters. No automatic merge; boilerplate can create false positives and low-text/scanned documents remain unverified.",
        "final_selection_frozen": False, "model_outputs_used": False,
        "source_semantics_verified": False, "duration_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("metadata_question_walk.jsonl", metadata_walk), ("technical_question_walk.jsonl", question_walk)):
        path = output / filename
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
        result["bindings"][filename + "_sha256"] = sha256_file(path)
    (output / "technical_census.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def render_audit(census_path: Path, assets: Path, output: Path, pdftoppm: Path,
                 limit: int = 800, start: int = 0) -> dict:
    """Render a bounded, ranked technical audit pack, not a final cohort."""
    from PIL import Image
    import os

    census_path, assets, output, pdftoppm = [Path(p).resolve() for p in (census_path, assets, output, pdftoppm)]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Audit-render output must be absent or empty")
    if type(limit) is not int or limit < 1 or type(start) is not int or start < 0:
        raise ValueError("Positive limit and nonnegative starting rank are required")
    census = json.loads(census_path.read_text(encoding="utf-8"))
    if (census["stage"] != "source_only_technical_census"
            or census["bindings"]["adapter_sha256"] != sha256_file(Path(__file__))
            or census["model_outputs_used"] is not False
            or census["final_selection_frozen"] is not False):
        raise ValueError("Census identity or preparation stage differs")
    asset_index, extraction = _asset_index(assets)
    if extraction != census["extraction"]:
        raise ValueError("Extracted source provenance differs from census")
    new_rows = [row for row in census["candidates"] if not row["prior_engineering_cluster"]]
    selected = new_rows[start:start + limit]
    if start == 0:
        selected += [row for row in census["candidates"] if row["prior_engineering_cluster"]]
    selected.sort(key=lambda row: row["source_rank"])
    if not selected:
        raise ValueError("No technical candidates in this audit range")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir()
    (output / "contexts").mkdir()
    version = subprocess.run([str(pdftoppm), "-v"], check=True, capture_output=True, text=True)
    version_text = (version.stdout + version.stderr).strip().splitlines()[0]
    manifests, audits, failures = [], [], []
    for index, candidate in enumerate(selected):
        doc_id, page_index = candidate["doc_id"], candidate["answer_page_index"]
        pdf_asset = asset_index[(doc_id, "pdf")]
        _read_bound_asset(pdf_asset)
        image_name = f"{doc_id}_p{page_index}"
        prefix = Path("images") / image_name
        source_pdf = os.path.relpath(pdf_asset["path"], output)
        began = time.perf_counter()
        try:
            subprocess.run([str(pdftoppm), "-f", str(page_index + 1), "-l", str(page_index + 1),
                            "-singlefile", "-r", "200", "-png", source_pdf, str(prefix)],
                           cwd=output, check=True, capture_output=True, timeout=180)
            image_path = output / (str(prefix) + ".png")
            with Image.open(image_path) as raw_image:
                image = raw_image.convert("RGB")
                if list(image.size) != candidate["pixel_dimensions"]:
                    raise Ineligible("actual_raster_dimensions_differ_from_verified_geometry")
                if raw_image.mode != "RGB":
                    image.save(image_path)
                roi = candidate["roi_pixels"]
                if not contains([0, 0, *image.size], roi):
                    raise Ineligible("roi_outside_actual_render")
                image.crop(tuple(roi)).save(output / "contexts" / f"{image_name}.png")
            image_sha = sha256_file(image_path)
            audit = {
                **candidate, "source_image_path": (prefix.as_posix() + ".png"),
                "context_image_path": f"contexts/{image_name}.png",
                "image_sha256": image_sha, "actual_render_size": candidate["pixel_dimensions"],
                "render_elapsed_s": time.perf_counter() - began,
                "census_sha256": sha256_file(census_path),
                "model_prompt_contract": "Only original question and selected page/crop pixels; no answers, OCR, coordinates, URL or audit text.",
                "full_page_semantic_review_required": True,
                "source_semantic_audit_status": "pending",
            }
            row = {
                "example_id": candidate["example_id"], "image_id": f"{doc_id}:page{page_index}",
                "source_id": candidate["source_cluster_id"], "source_cluster_id": candidate["source_cluster_id"],
                "doc_id": doc_id, "question_uid": candidate["question_uid"],
                "question": candidate["question"], "answer": candidate["answer"],
                "original_answers": candidate["original_answers"],
                "original_answer_variants": candidate["original_answer_variants"],
                "ocr_supported_answers": candidate["ocr_supported_answers"],
                "validated_primary_answers": [], "source_semantic_audit_status": "pending",
                "image_path": prefix.as_posix() + ".png", "image_sha256": image_sha,
                "rendered_png_sha256": image_sha, "pixel_dimensions": candidate["pixel_dimensions"],
                "width": candidate["pixel_dimensions"][0], "height": candidate["pixel_dimensions"][1],
                "render_dpi": 200, "render_mode": "RGB", "pdf_page_count": candidate["pdf_page_count"],
                "answer_page_index": page_index, "answer_page_privileged": True,
                "roi_pixels": candidate["roi_pixels"], "roi_privileged": True,
                "roi_provenance_sha256": canonical_hash(audit),
                "source_pdf_sha256": candidate["source_pdf_sha256"],
                "source_ocr_sha256": candidate["source_ocr_sha256"],
                "source_azure_original_sha256": candidate["source_azure_original_sha256"],
                "source_split": "train", "split": "replication",
                "cohort": candidate["cohort_reservation"], "source_rank": candidate["source_rank"],
                "lineage_verified": candidate["lineage_verified"], "source_urls": candidate["source_urls"],
                "source_cluster_doc_ids": candidate["source_cluster_doc_ids"],
                "prior_engineering_cluster": candidate["prior_engineering_cluster"],
                "near_duplicate_review_flags": candidate["near_duplicate_review_flags"],
                "partial_gold_word_coverage": any(not word["full_coverage"] for word in candidate["canonical_match"]["support"]),
            }
            manifests.append(row)
            audits.append(audit)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Ineligible, OSError) as error:
            failures.append({"example_id": candidate["example_id"], "source_cluster_id": candidate["source_cluster_id"],
                             "reason": str(error)[:500], "error_type": type(error).__name__,
                             "deterministic_source_exclusion": isinstance(error, Ineligible),
                             "infrastructure_resolution_required": not isinstance(error, Ineligible)})
        if (index + 1) % 25 == 0:
            print(f"Rendered source-audit pages: {index + 1}/{len(selected)}", flush=True)
    manifest_path, audit_path = output / "candidate_manifest.jsonl", output / "source_audit.jsonl"
    for path, rows in ((manifest_path, manifests), (audit_path, audits)):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    result = {
        "stage": "source_only_candidate_audit_pack", "created_utc": datetime.now(timezone.utc).isoformat(),
        "census_sha256": sha256_file(census_path), "adapter_sha256": sha256_file(Path(__file__)),
        "candidate_manifest_sha256": sha256_file(manifest_path), "source_audit_sha256": sha256_file(audit_path),
        "renderer": {"version": version_text, "executable_sha256": sha256_file(pdftoppm), "dpi": 200, "page_box": "MediaBox"},
        "new_candidate_start": start, "new_candidate_limit": limit, "attempted": len(selected),
        "rendered": len(manifests), "failures": failures,
        "final_selection_frozen": False, "model_outputs_used": False,
    }
    (output / "preparation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def validate_candidate_binding(row: dict, audit: dict, candidate: dict) -> None:
    """Cross-artifact equality is required in addition to hashes of each file."""
    fields = ("example_id", "source_id", "source_cluster_id", "source_cluster_doc_ids", "doc_id",
              "question_uid", "question", "answer", "original_answers", "original_answer_variants",
              "ocr_supported_answers", "answer_page_index", "answer_page_privileged", "roi_privileged",
              "roi_pixels", "pixel_dimensions", "pdf_page_count", "source_pdf_sha256", "source_ocr_sha256",
              "source_azure_original_sha256", "source_rank", "lineage_verified", "source_urls",
              "prior_engineering_cluster", "near_duplicate_review_flags")
    for field in fields:
        if field not in row or field not in audit or field not in candidate or not (row[field] == audit[field] == candidate[field]):
            raise ValueError(f"Candidate/source audit/census disagree: {field}")
    if canonical_hash({key: value for key, value in candidate.items() if key != "technical_provenance_sha256"}) != candidate["technical_provenance_sha256"]:
        raise ValueError("Technical candidate provenance hash differs")
    if (row["validated_primary_answers"] != [] or row["source_semantic_audit_status"] != "pending"
            or row["cohort"] != candidate["cohort_reservation"]
            or row["image_path"] != audit["source_image_path"]
            or row["image_sha256"] != audit["image_sha256"]
            or row["image_sha256"] != row["rendered_png_sha256"]
            or row["pixel_dimensions"] != audit["actual_render_size"]
            or [row["width"], row["height"]] != row["pixel_dimensions"]):
        raise ValueError("Rendered candidate metadata differs from its source audit")
    expected_partial = any(not word["full_coverage"] for word in candidate["canonical_match"]["support"])
    if row["partial_gold_word_coverage"] is not expected_partial:
        raise ValueError("Partial word-coverage flag differs")


def validate_render_failure(failure: dict, candidate: dict) -> None:
    if (failure.get("example_id") != candidate["example_id"]
            or failure.get("source_cluster_id") != candidate["source_cluster_id"]):
        raise ValueError("Render failure identity differs from ranked census source")
    if failure.get("deterministic_source_exclusion") is True:
        if (failure.get("error_type") != "Ineligible"
                or failure.get("infrastructure_resolution_required") is not False
                or failure.get("reason") not in {
                    "actual_raster_dimensions_differ_from_verified_geometry",
                    "roi_outside_actual_render"}):
            raise ValueError("Undeclared deterministic render exclusion")
    elif failure.get("infrastructure_resolution_required") is not True:
        raise ValueError("Unclassified renderer failure")


def validate_source_review(row: dict, review: dict) -> None:
    """Finalization guard: source approval cannot rewrite a reference."""
    if review.get("example_id") != row["example_id"] or review.get("source_cluster_id") != row["source_cluster_id"]:
        raise ValueError("Semantic review identity mismatch")
    if review.get("source_audit_sha256") != row["roi_provenance_sha256"]:
        raise ValueError("Semantic review is not bound to the rendered source audit")
    if type(review.get("semantic_approved")) is not bool:
        raise ValueError("Semantic review lacks an explicit approval decision")
    for field in ("rationale", "reviewer", "reviewed_at_utc"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError(f"Source review requires {field} for approval or rejection")
    try:
        reviewed = datetime.fromisoformat(review["reviewed_at_utc"].replace("Z", "+00:00"))
        if reviewed.tzinfo is None:
            raise ValueError("Naive review timestamp")
    except ValueError as error:
        raise ValueError("Source review requires a timezone-aware timestamp") from error
    if review.get("model_outputs_used") is not False:
        raise ValueError("Review must explicitly be source-only")
    if review.get("semantic_approved") is not True:
        return
    if review.get("full_page_visual_check") is not True:
        raise ValueError("An approved source requires a full-page visual check")
    if review.get("roi_context_sufficient") is not True:
        raise ValueError("An approved source requires sufficient question/ROI context")
    if review.get("duplicate_identity_resolved") is not True:
        raise ValueError("An approved source requires resolved identity/duplicate review")
    if row.get("partial_gold_word_coverage") and review.get("partial_word_coverage_checked") is not True:
        raise ValueError("Partial OCR-word support requires explicit visual review")
    approved = review.get("validated_primary_answers")
    original = row["original_answers"] + row["original_answer_variants"]
    if (not isinstance(approved, list) or not approved or any(not isinstance(v, str) or not v.strip() for v in approved)
            or row["original_answers"][0] not in approved
            or any(value not in original or value not in row["ocr_supported_answers"] for value in approved)):
        raise ValueError("Approved references must retain canonical gold and be source-supported original strings")
    if not isinstance(review.get("rationale"), str) or not review["rationale"].strip():
        raise ValueError("Source approval requires a recorded rationale")


def finalize(census_path: Path, audit_dirs: list[Path], review_path: Path,
             output: Path, main_count: int = 660) -> dict:
    """Freeze only the predeclared source-reviewed target; never lower its N."""
    import shutil

    census_path, review_path, output = [Path(p).resolve() for p in (census_path, review_path, output)]
    if main_count != 660:
        raise ValueError("The approved prospective main target is660; a new protocol is required to change it")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Final output must be absent or empty")
    census = json.loads(census_path.read_text(encoding="utf-8"))
    if (census["bindings"]["adapter_sha256"] != sha256_file(Path(__file__))
            or census.get("stage") != "source_only_technical_census"
            or census.get("seed") != SEED or census.get("model_outputs_used") is not False
            or census.get("final_selection_frozen") is not False
            or census.get("extraction", {}).get("verified_complete") is not True):
        raise ValueError("Census adapter differs")
    review_rows = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines()]
    if len({row["example_id"] for row in review_rows}) != len(review_rows):
        raise ValueError("Duplicate source review")
    reviews = {row["example_id"]: row for row in review_rows}
    census_candidates = {row["example_id"]: row for row in census["candidates"]}
    rendered, render_failures = {}, {}
    bindings = {}
    for audit_dir in map(lambda p: Path(p).resolve(), audit_dirs):
        prep = json.loads((audit_dir / "preparation.json").read_text(encoding="utf-8"))
        manifest_path, audit_path = audit_dir / "candidate_manifest.jsonl", audit_dir / "source_audit.jsonl"
        if (prep["census_sha256"] != sha256_file(census_path)
                or prep["adapter_sha256"] != sha256_file(Path(__file__))
                or prep["candidate_manifest_sha256"] != sha256_file(manifest_path)
                or prep["source_audit_sha256"] != sha256_file(audit_path)):
            raise ValueError("Candidate audit-pack binding differs")
        if len({failure["example_id"] for failure in prep["failures"]}) != len(prep["failures"]):
            raise ValueError("Duplicate render-failure IDs within audit pack")
        for failure in prep["failures"]:
            if failure["example_id"] not in census_candidates:
                raise ValueError("Render failure is absent from census")
            validate_render_failure(failure, census_candidates[failure["example_id"]])
            if failure["example_id"] in render_failures and render_failures[failure["example_id"]] != failure:
                raise ValueError("Conflicting repeated render failure")
            render_failures[failure["example_id"]] = failure
        audit_rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        audit_ids = [row["example_id"] for row in audit_rows]
        manifest_ids = [row["example_id"] for row in manifest_rows]
        if (len(set(audit_ids)) != len(audit_ids) or len(set(manifest_ids)) != len(manifest_ids)
                or set(audit_ids) != set(manifest_ids)):
            raise ValueError("Source audit and manifest IDs must be unique and equal")
        if len(manifest_rows) != prep["rendered"] or len(manifest_rows) + len(prep["failures"]) != prep["attempted"]:
            raise ValueError("Audit-pack render/failure counts differ")
        audit_map = {row["example_id"]: row for row in audit_rows}
        for row in manifest_rows:
            if row["example_id"] in rendered:
                raise ValueError("Overlapping audit packs are not accepted")
            if canonical_hash(audit_map[row["example_id"]]) != row["roi_provenance_sha256"]:
                raise ValueError("Candidate ROI audit hash differs")
            if row["example_id"] not in census_candidates:
                raise ValueError("Rendered candidate is absent from source census")
            validate_candidate_binding(row, audit_map[row["example_id"]], census_candidates[row["example_id"]])
            relative = Path(row["image_path"])
            image_path = (audit_dir / relative).resolve()
            if relative.is_absolute() or relative.drive or not image_path.is_relative_to(audit_dir) or sha256_file(image_path) != row["image_sha256"]:
                raise ValueError("Rendered source image differs")
            rendered[row["example_id"]] = (row, image_path, audit_map[row["example_id"]])
        bindings[str(audit_dir.name)] = sha256_file(audit_dir / "preparation.json")
    selected_main, selected_engineering, exclusions = [], [], []
    for candidate in census["candidates"]:
        key = candidate["example_id"]
        engineering = candidate["cohort_reservation"] == "engineering"
        if not engineering and len(selected_main) >= 660:
            continue
        if key not in rendered and key in render_failures:
            if render_failures[key].get("deterministic_source_exclusion") is not True:
                raise ValueError(f"Earlier renderer infrastructure failure requires resolution: {key}")
            exclusions.append({"example_id": key, "source_cluster_id": candidate["source_cluster_id"],
                               "reason": render_failures[key], "whole_cluster_excluded": True,
                               "stage": "source_render_failure"})
            continue
        if key not in rendered or key not in reviews:
            raise ValueError(f"Earlier-ranked source is not rendered/reviewed: {key}")
        row, image_path, audit = rendered[key]
        review = reviews[key]
        validate_source_review(row, review)
        if review["semantic_approved"] is not True:
            exclusions.append({"example_id": key, "source_cluster_id": row["source_cluster_id"],
                               "reason": review.get("rationale"), "whole_cluster_excluded": True})
            continue
        approved = {**row, "validated_primary_answers": review["validated_primary_answers"],
                    "source_semantic_audit_status": "approved",
                    "source_semantic_review_sha256": canonical_hash(review),
                    "cohort": "engineering" if engineering else "main"}
        (selected_engineering if engineering else selected_main).append((approved, image_path, audit, review))
    if len(selected_main) != 660:
        raise ValueError(f"Only{len(selected_main)} source-reviewed main clusters; the target is660. No lower-N manifest is written.")
    if len({entry[0]["source_cluster_id"] for entry in selected_main + selected_engineering}) != len(selected_main + selected_engineering):
        raise ValueError("A source cluster is shared across final rows")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir()
    final_audits = []
    for cohort, entries in (("main", selected_main), ("engineering", selected_engineering)):
        rows = []
        for row, image_path, audit, review in entries:
            target = output / row["image_path"]
            shutil.copyfile(image_path, target)
            if sha256_file(target) != row["image_sha256"]:
                raise ValueError("Final source-image copy differs")
            rows.append(row)
            final_audits.append({"example_id": row["example_id"], "technical": audit, "semantic": review})
        (output / f"{cohort}_manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    audit_path = output / "source_audit.jsonl"
    audit_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final_audits), encoding="utf-8", newline="\n")
    result = {
        "stage": "source_reviewed_replication_cohort", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "main_count": len(selected_main), "engineering_count": len(selected_engineering),
        "engineering_source_reservations": census["engineering_reserved_source_clusters"],
        "census_sha256": sha256_file(census_path), "review_sha256": sha256_file(review_path),
        "adapter_sha256": sha256_file(Path(__file__)), "audit_pack_sha256": bindings,
        "main_manifest_sha256": sha256_file(output / "main_manifest.jsonl"),
        "engineering_manifest_sha256": sha256_file(output / "engineering_manifest.jsonl"),
        "source_audit_sha256": sha256_file(audit_path), "source_only_exclusions": exclusions,
        "canonical_references_unchanged": True, "model_outputs_used": False,
        "final_selection_frozen": True, "answer_page_privileged": True, "roi_privileged": True,
    }
    (output / "selection_metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    census = commands.add_parser("census")
    census.add_argument("--metadata", type=Path, required=True)
    census.add_argument("--lineage", type=Path, required=True)
    census.add_argument("--assets", type=Path, required=True)
    census.add_argument("--tatdqa-raw", type=Path)
    census.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("render-audit")
    render.add_argument("--census", type=Path, required=True)
    render.add_argument("--assets", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--pdftoppm", type=Path, required=True)
    render.add_argument("--limit", type=int, default=800)
    render.add_argument("--start", type=int, default=0)
    final = commands.add_parser("finalize")
    final.add_argument("--census", type=Path, required=True)
    final.add_argument("--audit-dir", type=Path, action="append", required=True)
    final.add_argument("--reviews", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "census":
        result = technical_census(args.metadata, args.lineage, args.assets, args.output, args.tatdqa_raw)
        print(json.dumps(result["counts"], indent=2))
    elif args.command == "render-audit":
        result = render_audit(args.census, args.assets, args.output, args.pdftoppm, args.limit, args.start)
        print(json.dumps({"rendered": result["rendered"], "failures": len(result["failures"])}, indent=2))
    else:
        result = finalize(args.census, args.audit_dir, args.reviews, args.output)
        print(json.dumps({"main": result["main_count"], "engineering": result["engineering_count"]}, indent=2))


if __name__ == "__main__":
    main()
