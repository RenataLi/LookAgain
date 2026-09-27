"""Prepare an annotation-privileged, outcome-independent evidence-ROI pilot.

Requires Pillow and pypdf. No model, run records, or predictions are loaded.
Original v3 images, references and manifests remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import time
import zipfile

from PIL import Image

PINS = {
    "train_json_sha256": "3025b4ca9c3c87f5ecd2c504371e1ee92929ef1f31d8a3ac38bc3a5376582fab",
    "train_zip_sha256": "412dde804ec9d5888b888a9d7179f2157bc2679b92379f9804e6f2062e7237d9",
    "source_main_manifest_sha256": "757c3a4d024ac726f701c8732424e3457bfb292d224f67e2de22201ac641adc5",
    "source_smoke_manifest_sha256": "ac4f35d9edf4c6e6a7b70225668fc89172a618f3eaecaa661009269c2c109564",
    "source_quality_mask_sha256": "a621e6124f866ec4a1d0b882dc03cd5a10fe3bcfd8e567574c65934c76f9db90",
}


class Ineligible(ValueError):
    """A prespecified source-only exclusion, distinct from damaged provenance."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def normalize_span(text):
    return " ".join(text.casefold().split())


def valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(x) in (int, float) and math.isfinite(x) for x in box)
            and box[0] < box[2] and box[1] < box[3])


def contained(inner, outer):
    return valid_box(inner) and valid_box(outer) and outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def align_words(text, word_list):
    """Exact monotonic token positions, allowing only whitespace between tokens."""
    if not isinstance(text, str) or not isinstance(word_list, list) or not word_list:
        raise Ineligible("missing_word_geometry")
    spans, cursor = [], 0
    for word in word_list:
        if not isinstance(word, str) or not word:
            raise Ineligible("invalid_word_text")
        start = text.find(word, cursor)
        if start < 0 or text[cursor:start].strip():
            raise Ineligible("word_text_alignment_failure")
        spans.append([start, start + len(word)])
        cursor = start + len(word)
    if text[cursor:].strip():
        raise Ineligible("word_text_alignment_failure")
    return spans


def evidence_words(question, ocr):
    """Deduplicate annotation triples; select full word boxes overlapping offsets."""
    pages = ocr.get("pages")
    if not isinstance(pages, list) or len(pages) != 1:
        raise Ineligible("non_single_page_ocr")
    page = pages[0]
    page_box = page.get("bbox")
    if not valid_box(page_box):
        raise Ineligible("invalid_ocr_page_box")
    block_list = page.get("blocks", [])
    blocks = {b["uuid"]: b for b in block_list if "uuid" in b}
    if len(blocks) != len(block_list):
        raise Ineligible("missing_or_duplicate_block_uuid")
    mappings = question.get("block_mapping")
    if not isinstance(mappings, list) or not mappings:
        raise Ineligible("missing_mapping")
    seen, evidence, words, duplicate_count = set(), [], [], 0
    for mapping in mappings:
        if not isinstance(mapping, dict) or not mapping:
            raise Ineligible("invalid_mapping_structure")
        for uid, offsets in mapping.items():
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets):
                raise Ineligible("invalid_offsets")
            key = uid, *offsets
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            if uid not in blocks:
                raise Ineligible("missing_mapped_block")
            block = blocks[uid]
            text = block.get("text", "")
            if not (0 <= offsets[0] < offsets[1] <= len(text)):
                raise Ineligible("invalid_offsets")
            word_list = block.get("words", {}).get("word_list", [])
            boxes = block.get("words", {}).get("bbox_list", [])
            if len(word_list) != len(boxes):
                raise Ineligible("word_box_count_mismatch")
            spans = align_words(text, word_list)
            selected = [i for i, (start, end) in enumerate(spans) if start < offsets[1] and end > offsets[0]]
            if not selected:
                raise Ineligible("no_selected_words")
            chosen = []
            for i in selected:
                if not contained(boxes[i], page_box):
                    raise Ineligible("selected_word_box_outside_page")
                chosen.append({"word": word_list[i], "bbox": boxes[i], "char_span": spans[i]})
            fragments = [text[start:offsets[0]] for start, end in spans if start < offsets[0] < end]
            fragments += [text[offsets[1]:end] for start, end in spans if start < offsets[1] < end]
            unsafe = any(any(c.isalnum() or c in "-\u2010\u2011" for c in fragment) for fragment in fragments)
            evidence.append({"block_uid": uid, "block_bbox": block.get("bbox"), "block_text": text,
                "offsets": offsets, "mapped_text": text[offsets[0]:offsets[1]], "selected_words": chosen,
                "omitted_boundary_fragments": fragments, "unsafe_boundary_cut": unsafe})
            words.extend(chosen)
    selected_text = " ".join(item["mapped_text"] for item in evidence)
    return {"ocr_page_bbox": page_box, "mappings": evidence, "selected_words": words,
            "selected_text": selected_text, "duplicate_mappings_removed": duplicate_count,
            "unsafe_boundary_cut": any(item["unsafe_boundary_cut"] for item in evidence)}


def validate_pdf_geometry(geometry, page_bbox):
    media, crop = geometry["mediabox"], geometry["cropbox"]
    if geometry["page_count"] != 1:
        raise Ineligible("non_single_page_pdf")
    if geometry["rotation"] != 0:
        raise Ineligible("unsupported_pdf_rotation")
    if geometry["user_unit"] != 1:
        raise Ineligible("unsupported_pdf_user_unit")
    if not contained(crop, media):
        raise Ineligible("unsupported_cropbox_outside_mediabox")
    if not valid_box(page_bbox):
        raise Ineligible("invalid_ocr_page_box")
    ow, oh = page_bbox[2] - page_bbox[0], page_bbox[3] - page_bbox[1]
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    # Accommodate at most one raster-pixel rounding error on each OCR axis.
    if abs(ow * ch - oh * cw) > cw + ch:
        raise Ineligible("unreconciled_ocr_cropbox_aspect")


def map_ocr_box(box, page_bbox, pdf_geometry, render_size):
    """OCR top-left xyxy over CropBox -> source MediaBox raster, outward-rounded."""
    validate_pdf_geometry(pdf_geometry, page_bbox)
    if not contained(box, page_bbox):
        raise Ineligible("evidence_box_outside_ocr_page")
    width, height = render_size
    if any(type(v) is not int or v <= 0 for v in render_size):
        raise ValueError("Invalid rendered dimensions")
    media, crop = pdf_geometry["mediabox"], pdf_geometry["cropbox"]
    mx, my = width / (media[2] - media[0]), height / (media[3] - media[1])
    sx = (crop[2] - crop[0]) / (page_bbox[2] - page_bbox[0]) * mx
    sy = (crop[3] - crop[1]) / (page_bbox[3] - page_bbox[1]) * my
    dx, dy = (crop[0] - media[0]) * mx, (media[3] - crop[3]) * my
    result = [math.floor(dx + (box[0] - page_bbox[0]) * sx), math.floor(dy + (box[1] - page_bbox[1]) * sy),
              math.ceil(dx + (box[2] - page_bbox[0]) * sx), math.ceil(dy + (box[3] - page_bbox[1]) * sy)]
    if not contained(result, [0, 0, width, height]):
        raise Ineligible("mapped_box_outside_render")
    return result, {"scale_xy": [sx, sy], "offset_xy": [dx, dy],
        "pdf_to_render_scale_xy": [mx, my], "ocr_origin_xy": page_bbox[:2],
        "rounding": "floor left/top; ceil right/bottom", "coordinate_frame": "OCR CropBox top-left to source MediaBox raster top-left"}


def padded_roi(word_boxes, page_bbox, pdf_geometry, render_size):
    """Union, two median word-height margins, minimum256-square, shifted in frame."""
    if not word_boxes:
        raise Ineligible("no_selected_words")
    raw = [min(b[0] for b in word_boxes), min(b[1] for b in word_boxes), max(b[2] for b in word_boxes), max(b[3] for b in word_boxes)]
    box, transform = map_ocr_box(raw, page_bbox, pdf_geometry, render_size)
    median_height = statistics.median((b[3] - b[1]) * transform["scale_xy"][1] for b in word_boxes)
    margin = 2 * median_height
    width, height = render_size
    if min(width, height) < 256:
        raise Ineligible("source_below_minimum_roi_dimensions")
    rw = min(width, max(256, math.ceil(box[2] - box[0] + 2 * margin)))
    rh = min(height, max(256, math.ceil(box[3] - box[1] + 2 * margin)))
    left = min(width - rw, max(0, math.floor((box[0] + box[2] - rw) / 2)))
    top = min(height - rh, max(0, math.floor((box[1] + box[3] - rh) / 2)))
    roi = [left, top, left + rw, top + rh]
    if not contained(box, roi):
        raise Ineligible("padded_roi_does_not_enclose_evidence")
    fraction = rw * rh / (width * height)
    if fraction > .25:
        raise Ineligible("roi_exceeds_quarter_page_area")
    return roi, {"raw_word_bbox_ocr": raw, "raw_word_bbox_pixels": box,
        "median_word_height_pixels": median_height, "context_margin_pixels": margin,
        "minimum_roi_side_pixels": 256, "roi_page_area_fraction": fraction,
        "window_rule": "Expand each side by two median selected-word heights; ceil dimensions; minimum256x256; shift inside page without blank padding.",
        "transform": transform}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def prepare(raw, source_dir, quality_mask, output):
    from pypdf import PdfReader  # Preparation-only dependency; geometry helpers need no PDF runtime.
    started = time.perf_counter()
    raw, source_dir, quality_mask, output = map(lambda p: Path(p).resolve(), (raw, source_dir, quality_mask, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be absent or empty; frozen panels are never overwritten")
    paths = {"train_json_sha256": raw / "tat_dqa_dataset_train.json", "train_zip_sha256": raw / "tat_dqa_train.zip",
        "source_main_manifest_sha256": source_dir / "main_manifest.jsonl", "source_smoke_manifest_sha256": source_dir / "smoke_manifest.jsonl",
        "source_quality_mask_sha256": quality_mask}
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    if hashes != PINS:
        raise ValueError("A frozen raw asset, source manifest, or quality mask differs from its pin")
    mask = json.loads(quality_mask.read_text(encoding="utf-8"))
    entries = json.loads(paths["train_json_sha256"].read_text(encoding="utf-8"))
    lookup = {(entry["doc"]["uid"], q["uid"]): (entry["doc"], q) for entry in entries for q in entry["questions"]}
    cohorts = {name: read_jsonl(source_dir / f"{name}_manifest.jsonl") for name in ("main", "smoke")}
    all_sources = cohorts["main"] + cohorts["smoke"]
    if len(all_sources) != 102 or len({s["source_id"] for s in all_sources}) != 102:
        raise ValueError("Unexpected or overlapping frozen source panel")
    prepared, audits, decisions = {"main": [], "smoke": []}, [], []
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir()
    with zipfile.ZipFile(paths["train_zip_sha256"]) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        for cohort, sources in cohorts.items():
            for source in sources:
                uid, qid = source["doc_uid"], source["question_uid"]
                if not re.fullmatch(r"[a-f0-9]{32}", uid):
                    raise ValueError("Unsafe document UID")
                doc, question = lookup[(uid, qid)]
                if (source["question"] != question["question"] or [source["answer"]] != question["answer"] or doc["source"] != source["source_id"]):
                    raise ValueError("Frozen manifest disagrees with source annotation")
                category = mask["categories_by_example_id"][source["example_id"]]
                decision = {"example_id": source["example_id"], "source_id": source["source_id"], "cohort": cohort,
                    "source_quality_category": category, "eligible": False, "first_exclusion_reason": None}
                try:
                    if category not in ("none", "mapping_only"):
                        raise Ineligible("prior_quality_" + category)
                    ocr_bytes = archive.read(f"train/{uid}.json")
                    evidence = evidence_words(question, json.loads(ocr_bytes))
                    if normalize_span(evidence["selected_text"]) != normalize_span(source["answer"]):
                        raise Ineligible("selected_text_reference_mismatch")
                    if evidence["unsafe_boundary_cut"]:
                        raise Ineligible("alphanumeric_or_hyphen_boundary_cut")
                    pdf_bytes = archive.read(f"train/{uid}.pdf")
                    if hashlib.sha256(pdf_bytes).hexdigest() != source["source_pdf_sha256"]:
                        raise ValueError("Frozen source PDF hash differs")
                    reader = PdfReader(io.BytesIO(pdf_bytes))
                    if len(reader.pages) != 1:
                        raise Ineligible("non_single_page_pdf")
                    page = reader.pages[0]
                    geometry = {"page_count": len(reader.pages), "mediabox": [float(x) for x in page.mediabox],
                        "cropbox": [float(x) for x in page.cropbox], "rotation": page.rotation, "user_unit": float(page.get("/UserUnit", 1))}
                    decision["pdf_geometry"] = geometry
                    relative = Path(source["image_path"])
                    image_path = (source_dir / relative).resolve()
                    if relative.is_absolute() or relative.drive or source_dir not in image_path.parents:
                        raise ValueError("Unsafe source image path")
                    if sha256_file(image_path) != source["image_sha256"]:
                        raise ValueError("Frozen source image differs")
                    with Image.open(image_path) as image:
                        image.load()
                        if image.mode != "RGB" or list(image.size) != source["pixel_dimensions"]:
                            raise ValueError("Frozen rendered image metadata differs")
                        size = list(image.size)
                    roi, construction = padded_roi([w["bbox"] for w in evidence["selected_words"]], evidence["ocr_page_bbox"], geometry, size)
                    audit = {"example_id": source["example_id"], "source_id": source["source_id"], "cohort": cohort,
                        "doc_uid": uid, "question_uid": qid, "question": source["question"], "reference_answer": source["answer"],
                        "source_quality_category": category, "source_manifest_sha256": hashes[f"source_{cohort}_manifest_sha256"],
                        "source_image_sha256": source["image_sha256"], "source_pdf_sha256": source["source_pdf_sha256"],
                        "source_ocr_sha256": hashlib.sha256(ocr_bytes).hexdigest(), "source_ocr_zip_member": f"train/{uid}.json",
                        "source_size": size, "pdf_geometry": geometry, "evidence": evidence,
                        "roi_pixels": roi, "construction": construction,
                        "privileged_annotation_roi": True, "model_outcomes_used": False,
                        "inference_input_contract": "Only original question and selected page pixels; audit text, offsets, reference, and OCR are never prompt content."}
                    target = output / "images" / f"{uid}.png"
                    shutil.copyfile(image_path, target)
                    if sha256_file(target) != source["image_sha256"]:
                        raise ValueError("Copied source image changed")
                    prepared[cohort].append({**source, "image_path": f"images/{uid}.png", "roi_pixels": roi,
                        "roi_provenance_sha256": canonical_hash(audit), "source_manifest_sha256": hashes[f"source_{cohort}_manifest_sha256"],
                        "roi_privileged": True, "source_quality_category": category})
                    audits.append(audit)
                    decision.update(eligible=True, roi_pixels=roi, roi_provenance_sha256=canonical_hash(audit))
                except Ineligible as error:
                    decision["first_exclusion_reason"] = str(error)
                decisions.append(decision)
    if len(prepared["main"]) != 85 or len(prepared["smoke"]) != 1:
        raise ValueError("The declared supported source regime no longer yields85 main+1 smoke")
    for cohort in ("main", "smoke"):
        write_jsonl(output / f"{cohort}_manifest.jsonl", prepared[cohort])
    write_jsonl(output / "roi_audit.jsonl", audits)
    counts = {}
    for cohort in ("main", "smoke"):
        rows = [r for r in decisions if r["cohort"] == cohort]
        counts[cohort] = {"original_sources": len(rows), "eligible_sources": sum(r["eligible"] for r in rows),
            "excluded_sources": sum(not r["eligible"] for r in rows),
            "first_exclusion_reasons": dict(Counter(r["first_exclusion_reason"] for r in rows if not r["eligible"])),
            "eligible_quality_categories": dict(Counter(r["source_quality_category"] for r in rows if r["eligible"]))}
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "stage": "evidence_availability_development",
        "selection_uses_model_outputs": False, "same_prior_development_reports": True, "held_out_claim": False,
        "privileged_annotation_roi": True, "source_images_copied_byte_identically": True,
        "preparation_script_sha256": sha256_file(__file__), "pins": hashes,
        "main_manifest_sha256": sha256_file(output / "main_manifest.jsonl"), "smoke_manifest_sha256": sha256_file(output / "smoke_manifest.jsonl"),
        "roi_audit_sha256": sha256_file(output / "roi_audit.jsonl"), "counts": counts, "decisions": decisions,
        "eligibility_order": ["prior none/mapping_only quality category", "deduplicated valid block mapping and exact word alignment",
            "casefold/whitespace selected-text equality to unchanged reference", "no alphanumeric/hyphen boundary fragment",
            "single PDF page, rotation0,UserUnit1, contained CropBox and OCR aspect reconciliation", "in-frame ROI with two median word-height margin, minimum256-square, at most quarter-page area"],
        "unsupported_geometry": "CropBox outside MediaBox is excluded; no intersection fallback. Other rotations/user units or unreconciled OCR aspect also fail closed.",
        "pdf_rendering": "Reuses frozen200DPI Poppler MediaBox renders; no rerender or upsampling during preparation.",
        "duration_s": time.perf_counter() - started}
    (output / "selection_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--quality-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = prepare(args.raw, args.source_dir, args.quality_mask, args.output)
    print(json.dumps({"counts": metadata["counts"], "main_manifest_sha256": metadata["main_manifest_sha256"],
        "smoke_manifest_sha256": metadata["smoke_manifest_sha256"], "roi_audit_sha256": metadata["roi_audit_sha256"]}, indent=2))
