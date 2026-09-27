"""Prepare an outcome-blind, report-disjoint TAT-DQA development panel.

Uses original one-page PDFs and OCR metadata, never the released 224px PNGs.
Evidence/context lives in a separate audit pack and is not model input.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import time
import zipfile

from PIL import Image
from pypdf import PdfReader

EXPECTED_JSON_SHA = "3025b4ca9c3c87f5ecd2c504371e1ee92929ef1f31d8a3ac38bc3a5376582fab"
EXPECTED_ZIP_SHA = "412dde804ec9d5888b888a9d7179f2157bc2679b92379f9804e6f2062e7237d9"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ordering(seed: int, value: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(), value


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def safe_member(uid: str, suffix: str) -> str:
    if not re.fullmatch(r"[a-fA-F0-9]{32}", uid):
        raise ValueError(f"Unexpected document UID: {uid}")
    return f"train/{uid}.{suffix}"


def question_failure(question: dict) -> str | None:
    answer = question.get("answer")
    if question.get("answer_type") != "span":
        return "not_single_span_type"
    if not isinstance(answer, list) or len(answer) != 1 or not isinstance(answer[0], str) or not answer[0].strip():
        return "not_one_nonempty_string_answer"
    if question.get("scale") != "":
        return "nonempty_scale"
    if question.get("req_comparison") is not False:
        return "requires_comparison"
    if len(answer[0]) > 160 or len(answer[0].split()) > 25:
        return "answer_exceeds_fixed_length_limit"
    return None


def audit_evidence(doc: dict, question: dict, ocr: dict, pdf_text: str) -> dict:
    blocks = {block["uuid"]: block for page in ocr["pages"] for block in page.get("blocks", []) if "uuid" in block}
    mapped = []
    missing = []
    invalid_offsets = []
    for mapping in question.get("block_mapping", []):
        for uid, offsets in mapping.items():
            block = blocks.get(uid)
            if block is None:
                missing.append(uid)
                continue
            text = block.get("text", "")
            valid = (isinstance(offsets, list) and len(offsets) == 2
                     and all(isinstance(value, int) for value in offsets)
                     and 0 <= offsets[0] <= offsets[1] <= len(text))
            if not valid:
                invalid_offsets.append({"block_uid": uid, "offsets": offsets})
            mapped.append({"block_uid": uid, "text": text, "bbox": block.get("bbox"),
                           "offsets": offsets, "offsets_valid": valid,
                           "mapped_text": text[offsets[0]:offsets[1]] if valid else None})
    answer = question["answer"][0]
    needle = normalize(answer)
    return {
        "example_id": f"tatdqa:{question['uid']}", "doc_source": doc["source"],
        "doc_uid": doc["uid"], "question_uid": question["uid"],
        "question": question["question"], "reference_answer": answer,
        "raw_document_annotation": doc, "raw_question_annotation": question,
        "mapped_evidence_blocks": mapped, "missing_block_uids": missing,
        "invalid_block_offsets": invalid_offsets,
        "reference_in_mapped_block_text_normalized": any(needle in normalize(block["text"]) for block in mapped),
        "reference_in_extracted_pdf_text_normalized": needle in normalize(pdf_text),
        "audit_flags_are_not_eligibility_filters": True,
        "manual_semantic_review_status": "pending; audit pack supplied for review",
        "inference_input": False,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def prepare(raw: Path, output: Path, pdftoppm: Path, main_count: int = 100,
            smoke_count: int = 2, seed: int = 20260925, dpi: int = 200) -> Path:
    started = time.perf_counter()
    raw, output, pdftoppm = raw.resolve(), output.resolve(), pdftoppm.resolve()
    if main_count < 1 or smoke_count < 1 or dpi != 200:
        raise ValueError("Positive cohort sizes and the fixed 200 DPI are required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be absent or empty; preparation never overwrites a frozen panel")
    annotation_path, zip_path = raw / "tat_dqa_dataset_train.json", raw / "tat_dqa_train.zip"
    annotation_sha, archive_sha = sha_file(annotation_path), sha_file(zip_path)
    if annotation_sha != EXPECTED_JSON_SHA or archive_sha != EXPECTED_ZIP_SHA:
        raise ValueError("Raw release hashes differ from the pinned official training assets")
    version_result = subprocess.run([str(pdftoppm), "-v"], capture_output=True, text=True, check=True)
    version = (version_result.stdout + version_result.stderr).strip().splitlines()[0]
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    total = main_count + smoke_count
    min_area = 2 * 1024 * 32 * 32
    output.mkdir(parents=True, exist_ok=True)
    for name in ("images", "pdfs", "pdf_text"):
        (output / name).mkdir()
    reasons = Counter()
    doc_cache = {}
    candidates = defaultdict(list)
    ocr_by_doc = {}
    selected = []
    source_walk = []
    pdf_screen_failures = []
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        if len(names) != len(archive.infolist()):
            raise ValueError("Duplicate ZIP entries are not accepted")
        # OCR page lists and question metadata are inspected for the complete release.
        for entry in data:
            doc = entry["doc"]
            ocr = json.loads(archive.read(safe_member(doc["uid"], "json")))
            pages = ocr.get("pages")
            if not isinstance(pages, list):
                raise ValueError("OCR JSON has no page list")
            if len(pages) != 1:
                reasons["questions_in_non_single_page_ocr_document"] += len(entry["questions"])
                continue
            ocr_by_doc[doc["uid"]] = ocr
            for question in entry["questions"]:
                failure = question_failure(question)
                if failure:
                    reasons[failure] += 1
                else:
                    candidates[doc["source"]].append((doc, question))
        for source in sorted(candidates, key=lambda value: ordering(seed, value)):
            considered = []
            choice = None
            for doc, question in sorted(candidates[source], key=lambda pair: ordering(seed, pair[1]["uid"])):
                uid = doc["uid"]
                if uid not in doc_cache:
                    pdf_bytes = archive.read(safe_member(uid, "pdf"))
                    reader = PdfReader(io.BytesIO(pdf_bytes))
                    page_count = len(reader.pages)
                    failure = None
                    text = ""
                    if page_count != 1:
                        failure = "pdf_not_single_page"
                    else:
                        page = reader.pages[0]
                        text = page.extract_text() or ""
                        width_pt, height_pt = float(page.mediabox.width), float(page.mediabox.height)
                        user_unit = float(page.get("/UserUnit", 1))
                        predicted = [math.ceil(width_pt * user_unit * dpi / 72), math.ceil(height_pt * user_unit * dpi / 72)]
                        if len(text.strip()) < 100:
                            failure = "pdf_extracted_text_under_100_chars"
                        elif min(predicted) < 32 or math.prod(predicted) < min_area:
                            failure = "predicted_200dpi_dimensions_below_threshold"
                    info = {"pdf_page_count": page_count, "ocr_page_count": 1,
                            "pdf_extracted_text_chars": len(text.strip()), "failure": failure,
                            "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(), "text": text}
                    if page_count == 1:
                        info.update(media_box_points=list(map(float, page.mediabox)),
                                    crop_box_points=list(map(float, page.cropbox)),
                                    pdf_rotation=int(page.get("/Rotate", 0)), pdf_user_unit=user_unit,
                                    predicted_200dpi_pixels=predicted)
                    doc_cache[uid] = info
                    if failure:
                        pdf_screen_failures.append({"doc_uid": uid, "source": source, "reason": failure})
                info = doc_cache[uid]
                considered.append({"doc_uid": uid, "question_uid": question["uid"], "technical_failure": info["failure"]})
                if not info["failure"]:
                    choice = (doc, question, info)
                    break
            source_walk.append({"source_id": source, "source_order_sha256": ordering(seed, source)[0],
                                "considered_questions": considered, "selected": choice is not None})
            if choice is not None:
                selected.append(choice)
            if len(selected) == total:
                break
        if len(selected) != total:
            raise ValueError(f"Only {len(selected)} eligible sources; need {total}")

        rows, audits = [], []
        for index, (doc, question, info) in enumerate(selected):
            uid = doc["uid"]
            pdf_path = output / "pdfs" / f"{uid}.pdf"
            pdf_path.write_bytes(archive.read(safe_member(uid, "pdf")))
            (output / "pdf_text" / f"{uid}.txt").write_text(info["text"], encoding="utf-8")
            render_started = time.perf_counter()
            # Relative ASCII input/output arguments also support Windows user paths.
            command = [str(pdftoppm), "-f", "1", "-singlefile", "-r", str(dpi), "-png",
                       f"pdfs/{uid}.pdf", f"images/{uid}"]
            subprocess.run(command, cwd=output, check=True, capture_output=True, timeout=120)
            image_path = output / "images" / f"{uid}.png"
            with Image.open(image_path) as image:
                dimensions = list(image.size)
                mode = image.mode
                if mode != "RGB":
                    rgb = image.convert("RGB")
                    rgb.save(image_path)
            if min(dimensions) < 32 or math.prod(dimensions) < min_area:
                raise ValueError(f"Actual rendered dimensions fail the frozen rule: {uid}")
            render_elapsed = time.perf_counter() - render_started
            cohort = "smoke" if index < smoke_count else "main"
            image_sha = sha_file(image_path)
            row = {
                "example_id": f"tatdqa:{question['uid']}", "image_id": uid,
                "source_id": doc["source"], "source_filename": doc["source"],
                "doc_uid": uid, "question_uid": question["uid"],
                "image_path": image_path.relative_to(output).as_posix(),
                "question": question["question"], "answer": question["answer"][0],
                "dataset": "TAT-DQA", "source_split": "train", "split": "development",
                "cohort": cohort, "selection_rank": index + 1,
                "image_sha256": image_sha, "rendered_png_sha256": image_sha,
                "source_pdf_sha256": info["pdf_sha256"],
                "pixel_dimensions": dimensions, "width": dimensions[0], "height": dimensions[1],
                "render_dpi": dpi, "render_mode": "RGB", "original_poppler_mode": mode,
                "render_elapsed_s": render_elapsed,
                "pdf_page_count": info["pdf_page_count"], "ocr_page_count": info["ocr_page_count"],
                "pdf_extracted_text_chars": info["pdf_extracted_text_chars"],
                "doc_page_metadata": doc["page"],
                "annotation_source_sha256": annotation_sha, "zip_source_sha256": archive_sha,
            }
            audit = audit_evidence(doc, question, ocr_by_doc[uid], info["text"])
            audit.update(cohort=cohort, selection_rank=index + 1,
                         pdf_metadata={key: value for key, value in info.items() if key != "text"},
                         rendered_dimensions=dimensions, image_sha256=image_sha,
                         pdf_text_path=f"pdf_text/{uid}.txt")
            rows.append(row)
            audits.append(audit)
            print(f"Rendered {index+1}/{total}: {cohort}, {dimensions[0]}x{dimensions[1]}", flush=True)
    main_rows = [row for row in rows if row["cohort"] == "main"]
    smoke_rows = [row for row in rows if row["cohort"] == "smoke"]
    assert len({row["source_id"] for row in rows}) == total
    main_path, smoke_path = output / "main_manifest.jsonl", output / "smoke_manifest.jsonl"
    write_jsonl(main_path, main_rows)
    write_jsonl(smoke_path, smoke_rows)
    write_jsonl(output / "annotation_audit.jsonl", audits)
    metadata = {
        "status": "selection_frozen_before_inference", "frozen_at_utc": utc_now(),
        "seed": seed, "main_count": main_count, "smoke_count": smoke_count,
        "sampling": "Sort reports by SHA256(seed:doc.source), then questions within each report by SHA256(seed:question.uid); take first technically eligible question per report. First two selected reports are smoke, following 100 are main.",
        "selection_uses_model_outcomes": False, "answer_context_match_is_eligibility": False,
        "selection_uses_answer_location_or_difficulty": False,
        "eligibility": "Actual PDF and OCR each one page; single nonempty string span; empty scale; req_comparison=false; answer <=160 chars and <=25 whitespace words; PDF extractable text >=100 stripped chars; actual full-page 200DPI RGB raster area >=2097152 and both dimensions >=32.",
        "vector_text_screen_limitation": "Extractable text is a technical screen, not proof of visual fidelity or label correctness.",
        "released_224px_pngs_used": False,
        "source_annotation_sha256": annotation_sha, "source_zip_sha256": archive_sha,
        "script_sha256": sha_file(Path(__file__)),
        "renderer": {"path": str(pdftoppm), "version": version, "executable_sha256": sha_file(pdftoppm),
                     "dpi": dpi, "page_box": "full MediaBox (pdftoppm default; no -cropbox)", "output_mode": "RGB",
                     "arguments": "-f 1 -singlefile -r 200 -png pdfs/DOC.pdf images/DOC"},
        "preprocessing_elapsed_s": time.perf_counter() - started,
        "main_manifest_sha256": sha_file(main_path), "smoke_manifest_sha256": sha_file(smoke_path),
        "annotation_audit_sha256": sha_file(output / "annotation_audit.jsonl"),
        "counts": {"all_document_entries": len(data), "all_questions": sum(len(entry["questions"]) for entry in data),
                   "candidate_reports_after_ocr_and_question_filter": len(candidates),
                   "candidate_questions_after_ocr_and_question_filter": sum(map(len, candidates.values())),
                   "metadata_first_failure_counts": dict(reasons), "reports_considered": len(source_walk),
                   "pdf_documents_screened": len(doc_cache), "pdf_screen_failures": len(pdf_screen_failures),
                   "selected_reports": total, "pdfs_extracted_and_rendered": total},
        "pdf_screen_failure_records": pdf_screen_failures, "source_selection_walk": source_walk,
        "audit_checks": {"all_selected_questions_have_evidence_record": len(audits) == total,
                         "reference_absent_from_mapped_blocks": sum(not a["reference_in_mapped_block_text_normalized"] for a in audits),
                         "reference_absent_from_pdf_text": sum(not a["reference_in_extracted_pdf_text_normalized"] for a in audits),
                         "records_with_missing_evidence_blocks": sum(bool(a["missing_block_uids"]) for a in audits),
                         "records_with_invalid_offsets": sum(bool(a["invalid_block_offsets"]) for a in audits),
                         "manual_semantic_review": "Audit pack supplied; no claim of completed manual review by preparation script."},
    }
    (output / "selection_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# TAT-DQA V3 source and annotation review pack", "",
             "All 102 selected questions are listed below with unchanged references and released evidence blocks. "
             "This audit pack is separate from model inputs. Automatic text-presence and offset checks do not establish semantic correctness. "
             "No examples were removed based on these checks. Manual semantic review remains pending unless separately documented.", ""]
    for audit in audits:
        flags = []
        if not audit["reference_in_mapped_block_text_normalized"]:
            flags.append("reference not found by normalized substring in mapped blocks")
        if not audit["reference_in_extracted_pdf_text_normalized"]:
            flags.append("reference not found by normalized substring in extracted PDF text")
        if audit["missing_block_uids"]:
            flags.append("missing evidence block")
        if audit["invalid_block_offsets"]:
            flags.append("invalid evidence offset")
        lines += [f"## {audit['selection_rank']}. {audit['cohort']} - {audit['doc_source']}", "",
                  f"Question ID: `{audit['question_uid']}`; document: `{audit['doc_uid']}`.", "",
                  "**Question:** " + audit["question"], "", "**Reference:** " + audit["reference_answer"], "",
                  "**Automatic flags:** " + ("; ".join(flags) if flags else "none"), ""]
        for block in audit["mapped_evidence_blocks"]:
            lines += [f"Evidence block `{block['block_uid']}`; offsets `{block['offsets']}`:", "",
                      block["text"], "", "Mapped substring: " + str(block["mapped_text"]), ""]
    (output / "annotation_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("Selection frozen:", output / "selection_metadata.json", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdftoppm", type=Path, required=True)
    parser.add_argument("--main-count", type=int, default=100)
    parser.add_argument("--smoke-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.pdftoppm, args.main_count, args.smoke_count, args.seed)


if __name__ == "__main__":
    main()
