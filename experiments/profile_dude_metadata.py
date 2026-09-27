"""Offline, training-only DUDE metadata census; no PDF or model access.

Bounding-box values are profiled as supplied. No pixel scale, canvas size, global
page-index convention, source independence or annotation correctness is assumed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit
import unicodedata

PROJECT = Path(__file__).resolve().parents[1]
PINS = {
    "annotations": "4f883956ba02cb676022ec24878cdc415e759438e945c2d479bdae488873e2e8",
    "lineage": "fbe2888343b3b7f137201b679355a54035b70cebddba19ed0d0374807b2299e9",
    "prior_main": "757c3a4d024ac726f701c8732424e3457bfb292d224f67e2de22201ac641adc5",
    "prior_smoke": "ac4f35d9edf4c6e6a7b70225668fc89172a618f3eaecaa661009269c2c109564",
}
BOX_FIELDS = ("left", "top", "width", "height", "page")
TRACKING_KEYS = {"fbclid", "gclid"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_url(value: str) -> str | None:
    """Conservative duplicate screen; keep meaningful query and path case."""
    parts = urlsplit(value.strip())
    if parts.scheme.casefold() not in {"http", "https"} or not parts.hostname:
        return None
    host = parts.hostname.casefold()
    if host.startswith("www."):
        host = host[4:]
    port = parts.port
    if port is not None and port not in (80, 443):
        host += ":" + str(port)
    path = unicodedata.normalize("NFC", unquote(parts.path))
    query = [(key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True)
             if not key.casefold().startswith("utm_") and key.casefold() not in TRACKING_KEYS]
    return host + path.rstrip("/") + ("?" + urlencode(sorted(query)) if query else "")


def compact_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", value).casefold())


def short_extractive_failure(row: dict) -> str | None:
    if row.get("answer_type") != "extractive":
        return "not_extractive"
    answers = row.get("answers")
    if not isinstance(answers, list) or len(answers) != 1:
        return "not_one_answer"
    answer = answers[0]
    if not isinstance(answer, str) or not answer.strip():
        return "not_nonempty_string"
    if len(answer) > 160 or len(answer.split()) > 25:
        return "answer_exceeds_fixed_length_limit"
    return None


def inspect_boxes(value) -> dict:
    """Only validate the schema and nonnegative extent, never unknown canvases."""
    if not isinstance(value, list) or len(value) != 1:
        return {"valid": False, "reason": "not_one_bbox_group"}
    boxes = value[0]
    if not isinstance(boxes, list) or not boxes:
        return {"valid": False, "reason": "missing_box_list"}
    for box in boxes:
        if not isinstance(box, dict) or any(key not in box for key in BOX_FIELDS):
            return {"valid": False, "reason": "missing_box_fields"}
        if any(type(box[key]) not in (int, float) or not math.isfinite(box[key]) for key in BOX_FIELDS):
            return {"valid": False, "reason": "nonfinite_or_nonnumeric_box"}
        if any(type(box[key]) is not int for key in BOX_FIELDS):
            return {"valid": False, "reason": "noninteger_box"}
        if box["left"] < 0 or box["top"] < 0 or box["width"] <= 0 or box["height"] <= 0:
            return {"valid": False, "reason": "negative_origin_or_nonpositive_extent"}
        if box["page"] < 0:
            return {"valid": False, "reason": "negative_page"}
    pages = sorted({box["page"] for box in boxes})
    return {
        "valid": True, "reason": None, "box_count": len(boxes), "pages": pages,
        "same_annotated_page": len(pages) == 1,
        "union_xyxy_if_same_page": [
            min(box["left"] for box in boxes), min(box["top"] for box in boxes),
            max(box["left"] + box["width"] for box in boxes),
            max(box["top"] + box["height"] for box in boxes),
        ] if len(pages) == 1 else None,
    }


def quantiles(values: list[int]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    result = {}
    for label, probability in (("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95), ("max", 1)):
        position = (len(values) - 1) * probability
        lo, hi = math.floor(position), math.ceil(position)
        result[label] = values[lo] + (values[hi] - values[lo]) * (position - lo)
    return result


def profile(metadata: Path, lineage: Path, prior_main: Path, prior_smoke: Path,
            output: Path) -> dict:
    metadata, lineage, prior_main, prior_smoke, output = [
        Path(p).resolve() for p in (metadata, lineage, prior_main, prior_smoke, output)]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be absent or empty; existing reports are not overwritten")
    paths = {"annotations": metadata, "lineage": lineage,
             "prior_main": prior_main, "prior_smoke": prior_smoke}
    hashes = {}
    for key, path in paths.items():
        hashes[key] = sha256_file(path)
        if hashes[key] != PINS[key]:
            raise ValueError(f"Input differs from its frozen SHA256: {key}")
    data = json.loads(metadata.read_text(encoding="utf-8"))
    # Split filtering precedes every eligibility, bounding-box and lineage analysis.
    all_rows = data["data"]
    split_inventory = dict(Counter(row.get("data_split") for row in all_rows))
    train = [row for row in all_rows if row.get("data_split") == "train"]
    question_groups = defaultdict(list)
    for row in train:
        question_groups[row["questionId"]].append(row)
    duplicate_questions = []
    for question_id, rows in question_groups.items():
        if len(rows) > 1:
            if any(row != rows[0] for row in rows[1:]):
                raise ValueError("Conflicting duplicate training questionId")
            duplicate_questions.append({"question_id": question_id, "copies": len(rows),
                                        "answer_type": rows[0]["answer_type"], "identical": True})
    with lineage.open(encoding="utf-8-sig", newline="") as stream:
        lineage_rows = list(csv.DictReader(stream))
    prior = [json.loads(line) for path in (prior_main, prior_smoke)
             for line in path.read_text(encoding="utf-8").splitlines()]
    prior_sources = sorted({row["source_id"] for row in prior})
    if len(prior_sources) != 102:
        raise ValueError("Historical source exclusion must contain 102 reports")
    lineage_by_doc = defaultdict(list)
    for row in lineage_rows:
        identifier = row["url_md5 (filename)"].strip()
        if identifier:
            lineage_by_doc[identifier].append(row)
    failures, box_failures, page_frequency = Counter(), Counter(), Counter()
    box_count_frequency, group_lengths, field_sets = Counter(), Counter(), Counter()
    coordinate_values = defaultdict(list)
    candidates, short_rows = [], []
    train_docs = {row["docId"] for row in train}
    per_doc = {doc_id: {"doc_id": doc_id, "train_questions": 0,
                       "short_single_extractive_questions": 0,
                       "same_page_metadata_candidates": 0} for doc_id in train_docs}
    for row in train:
        doc_id = row["docId"]
        per_doc[doc_id]["train_questions"] += 1
        failure = short_extractive_failure(row)
        if failure:
            failures[failure] += 1
            continue
        short_rows.append(row)
        per_doc[doc_id]["short_single_extractive_questions"] += 1
        groups = row.get("answers_page_bounding_boxes")
        group_lengths[len(groups) if isinstance(groups, list) else "not_list"] += 1
        box_result = inspect_boxes(groups)
        if not box_result["valid"]:
            box_failures[box_result["reason"]] += 1
            continue
        boxes = groups[0]
        box_count_frequency[len(boxes)] += 1
        for box in boxes:
            field_sets[",".join(sorted(box))] += 1
            page_frequency[box["page"]] += 1
            for key in BOX_FIELDS:
                coordinate_values[key].append(box[key])
            coordinate_values["right"].append(box["left"] + box["width"])
            coordinate_values["bottom"].append(box["top"] + box["height"])
        if not box_result["same_annotated_page"]:
            box_failures["multiple_annotated_pages"] += 1
            continue
        per_doc[doc_id]["same_page_metadata_candidates"] += 1
        candidates.append({"doc_id": doc_id, "question_id": row["questionId"],
                           "annotated_page": box_result["pages"][0],
                           "box_count": box_result["box_count"],
                           "annotated_union_xyxy": box_result["union_xyxy_if_same_page"]})
    candidate_docs = {row["doc_id"] for row in candidates}
    raw_url_docs, normalized_url_docs = defaultdict(set), defaultdict(set)
    doccloud_docs = defaultdict(set)
    candidate_urls_by_doc = defaultdict(list)
    overlap_hits = []
    for doc_id in sorted(train_docs):
        for item in lineage_by_doc.get(doc_id, []):
            url = item["url"].strip()
            if not url:
                continue
            raw_url_docs[url].add(doc_id)
            key = normalized_url(url)
            if key:
                normalized_url_docs[key].add(doc_id)
                match = re.search(r"(?:^|\.)documentcloud\.org/documents/(\d+)", key)
                if match:
                    doccloud_docs[match[1]].add(doc_id)
            if doc_id in candidate_docs:
                candidate_urls_by_doc[doc_id].append(url)
            parts = urlsplit(url)
            basename = unquote(parts.path.rsplit("/", 1)[-1])
            label = compact_label(re.sub(r"\.pdf$", "", basename, flags=re.I))
            for source in prior_sources:
                old_label = compact_label(source[:-4])
                if label and label == old_label:
                    overlap_hits.append({"doc_id": doc_id, "source_id": source,
                                         "url": url, "kind": "normalized_url_basename_equals_historical_report"})
    candidate_with_lineage = candidate_docs & set(lineage_by_doc)
    candidate_with_url = set(candidate_urls_by_doc)
    for doc_id in per_doc:
        per_doc[doc_id]["lineage_rows"] = len(lineage_by_doc.get(doc_id, []))
    match_counts = {
        "training_docs_with_lineage_row": len(train_docs & set(lineage_by_doc)),
        "training_docs_without_lineage_row": len(train_docs - set(lineage_by_doc)),
        "candidate_docs_with_lineage_row": len(candidate_with_lineage),
        "candidate_docs_without_lineage_row": len(candidate_docs - set(lineage_by_doc)),
        "candidate_docs_with_url": len(candidate_with_url),
        "candidate_questions_with_lineage_row": sum(row["doc_id"] in lineage_by_doc for row in candidates),
    }
    result = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "metadata_only_source_feasibility",
        "dataset": {"dataset_name": data.get("dataset_name"), "dataset_version": data.get("dataset_version")},
        "bindings": {"input_sha256": hashes, "script_sha256": sha256_file(Path(__file__))},
        "source_urls": {
            "annotations": "https://zenodo.org/records/7763635/files/2023-03-23_DUDE_gt_test_PUBLIC.json?download=1",
            "lineage": "https://huggingface.co/datasets/jordyvl/DUDE_loader/resolve/b3662175d3b2482d711f18559b7acc2a5bccc600/licenses_metadata_datalineage.csv",
        },
        "split_policy": {
            "eligibility_split": "train", "split_row_inventory_only": split_inventory,
            "validation_test_eligibility_or_bbox_inspected": False,
            "note": "Validation/test rows are excluded before eligibility and geometry checks; only their split labels were counted.",
        },
        "training": {
            "questions": len(train), "doc_ids": len(train_docs),
            "unique_question_ids": len(question_groups),
            "duplicate_question_groups": duplicate_questions,
            "count_scope": "Observed metadata rows; identical duplicate abstractive rows are disclosed and do not enter the extractive pool.",
            "answer_type_counts": dict(Counter(row["answer_type"] for row in train)),
            "extractive_questions": sum(row.get("answer_type") == "extractive" for row in train),
            "short_single_extractive_questions": len(short_rows),
            "short_single_extractive_doc_ids": len({row["docId"] for row in short_rows}),
            "metadata_first_failure_counts": dict(failures),
            "same_page_metadata_candidate_questions": len(candidates),
            "same_page_metadata_candidate_doc_ids": len(candidate_docs),
            "same_page_candidates_with_lineage": match_counts,
        },
        "boxes": {
            "scope": "Only short single-answer extractive training questions.",
            "outer_group_length_counts": dict(group_lengths), "first_failure_counts": dict(box_failures),
            "valid_box_count_per_question_distribution": dict(box_count_frequency),
            "observed_box_field_sets": dict(field_sets),
            "observed_box_annotation_page_distribution": dict(sorted(page_frequency.items())),
            "same_page_candidate_question_page_distribution": dict(sorted(Counter(row["annotated_page"] for row in candidates).items())),
            "coordinate_quantiles": {key: quantiles(values) for key, values in coordinate_values.items()},
            "quantile_method": "Linear interpolation at probability*(n-1) on sorted observed values.",
            "coordinate_semantics": "left/top/width/height plus page, as supplied; no global unit, DPI, canvas size or index base inferred.",
            "canvas_bounds_verified": False, "pdf_page_counts_verified": False,
            "warning": "Finite integer coordinates and nonnegative extents do not establish that boxes fit a source page or locate the referenced answer.",
        },
        "lineage": {
            "csv_rows": len(lineage_rows),
            "nonempty_master_hash_rows": sum(bool(row["master_hash"].strip()) for row in lineage_rows),
            "matching_rule": "Exact docId equals CSV url_md5 (filename); empty master_hash cannot add aliases.",
            **match_counts,
            "exact_url_multi_doc_groups": [{"url": key, "doc_ids": sorted(value)} for key, value in raw_url_docs.items() if len(value) > 1],
            "normalized_url_multi_doc_groups": [{"url": key, "doc_ids": sorted(value)} for key, value in normalized_url_docs.items() if len(value) > 1],
            "documentcloud_numeric_id_multi_doc_groups": [{"documentcloud_id": key, "doc_ids": sorted(value)} for key, value in doccloud_docs.items() if len(value) > 1],
            "normalization": "Drop scheme and www; lowercase hostname; URL-decode/NFC path while preserving path case; drop fragment/tracking parameters; sort other query parameters.",
            "historical_tatdqa_report_count": len(prior_sources),
            "historical_tatdqa_filename_overlap_hits": overlap_hits,
            "overlap_limits": "URL basename comparisons cannot rule out report aliases, renamed downloads, near-duplicates, shared parent reports or TAT-DQA overlap. PDF bytes and content were not compared.",
            "license_limits": "CSV labels are metadata only; license/access/redistribution validity and current URLs were not verified.",
        },
        "per_training_doc_metadata_counts": sorted(per_doc.values(), key=lambda row: row["doc_id"]),
        "candidate_question_geometry": candidates,
        "feasibility": {
            "500_to_640_documents_metadata_headroom": len(candidate_docs) >= 640,
            "500_to_640_verified_independent_original_reports_available": None,
            "native_roi_readiness": False,
            "reason": "Potential metadata pool only. Original PDF availability, pages, scale/coordinate transform, text alignment, reference semantics, source identity and duplicates remain unverified.",
        },
        "published_count_note": "The downloaded metadata has 23736 training questions/2974 docIds; the paper's Table 2 reports 23728/3010. Preserve observed release counts; the version/revision discrepancy is not reconciled here.",
        "published_count_source": "https://arxiv.org/html/2305.08455v3#S3.T2",
        "next_checks": [
            "Freeze the source-only screening protocol and lineage identity/duplicate rules before choosing a panel.",
            "Confirm original-PDF availability and page count; do not equate one annotated answer page with a one-page document.",
            "Verify canvas units, DPI, page-index base, CropBox/MediaBox/rotation and box alignment on actual source pages.",
            "Audit answer/reference semantics and exact evidence support without model outcomes.",
            "Reserve source-inspected engineering examples separately if the future protocol chooses to do so; no selection is made here.",
        ],
        "selection_frozen": False, "random_seed_selected": None, "model_outputs_used": False,
        "pdf_files_read": False, "model_loaded": False,
    }
    if sum(failures.values()) + len(short_rows) != len(train):
        raise ValueError("Training metadata exclusions do not reconcile")
    if sum(box_failures.values()) + len(candidates) != len(short_rows):
        raise ValueError("Bounding-box exclusions do not reconcile")
    output.mkdir(parents=True, exist_ok=True)
    (output / "dude_metadata_feasibility.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "dude_metadata_feasibility.md").write_text(render_markdown(result), encoding="utf-8", newline="\n")
    return result


def render_markdown(result: dict) -> str:
    train, boxes, lineage = result["training"], result["boxes"], result["lineage"]
    return "\n".join([
        "# DUDE training metadata: conditional feasibility", "",
        "This census reads annotation and source-lineage metadata only. It does not read PDFs, call a model, "
        "select questions or assume a coordinate scale/page-index convention. Validation/test rows are removed "
        "before all eligibility and box checks; their split labels are counted only for inventory.", "",
        "| Training-only step | Questions | Document IDs |",
        "| --- | ---: | ---: |",
        f"| All training metadata | {train['questions']} | {train['doc_ids']} |",
        f"| Extractive answer type | {train['extractive_questions']} | — |",
        f"| One nonempty answer, ≤160 characters and 25 words | {train['short_single_extractive_questions']} | {train['short_single_extractive_doc_ids']} |",
        f"| Finite integer boxes, positive extents, nonnegative coordinates, one annotated answer page | {train['same_page_metadata_candidate_questions']} | {train['same_page_metadata_candidate_doc_ids']} |", "",
        "**These are document IDs and metadata candidates, not verified independent original reports or usable native ROIs.** "
        "One answer page does not establish a single-page original PDF. Bounding boxes can exceed 1000; "
        "normalizing them to a 1000-unit canvas would be unjustified.", "",
        "## Source identity and provenance", "",
        f"The lineage CSV directly matches {lineage['training_docs_with_lineage_row']} training docIds and leaves "
        f"{lineage['training_docs_without_lineage_row']} unmatched. Among same-page candidates, "
        f"{lineage['candidate_docs_with_lineage_row']} docIds have a lineage row and"
        f"{lineage['candidate_docs_without_lineage_row']} do not. All master_hash fields are empty.", "",
        f"Exact URL groups shared by multiple training docIds: {len(lineage['exact_url_multi_doc_groups'])}; "
        f"normalized URL groups: {len(lineage['normalized_url_multi_doc_groups'])}; "
        f"DocumentCloud numeric-ID groups: {len(lineage['documentcloud_numeric_id_multi_doc_groups'])}. "
        f"Normalized URL-basename matches to the 102 historical TAT-DQA report filenames: "
        f"{len(lineage['historical_tatdqa_filename_overlap_hits'])}. A zero-match count does not establish dataset independence; "
        "aliases, original-report grouping and PDF/content duplicates still need checking.", "",
        "The pinned [official annotation release](" + result["source_urls"]["annotations"] + ") and "
        "[author-maintained lineage CSV](" + result["source_urls"]["lineage"] + ") are bound by SHA256 in the JSON. "
        "The training count in this downloaded release differs from "
        "[the paper's Table 2](https://arxiv.org/html/2305.08455v3#S3.T2), which gives 23728 questions/3010 documents; "
        "this report retains the observed 23736/2974 and does not silently reconcile versions. "
        "License labels and URLs were profiled as metadata; access and redistribution rights were not revalidated.", "",
        "## Geometry limitations", "",
        "The JSON records box/group shape, first failures, count/page distributions and coordinate quantiles. "
        "Checks establish finite integer left/top/width/height/page values with positive sizes and nonnegative origins. "
        "Canvas bounds, DPI, global page-index base, PDF page counts, PDF boxes/rotation, answer-text alignment "
        "and annotation semantics remain unverified. No source crop is constructed.", "",
        f"Training metadata has{train['unique_question_ids']} unique question IDs in{train['questions']} rows. "
        f"The{len(train['duplicate_question_groups'])} duplicate-ID groups contain identical abstractive records; "
        "they do not enter the extractive candidate pool. Counts retain the observed release rows rather than silently deduplicating.", "",
        "## Reproduction", "",
        "From the repository root, place the two pinned files under data/dude_metadata or pass their paths explicitly:", "",
        "```text",
        "python experiments/profile_dude_metadata.py --metadata data/dude_metadata/2023-03-23_DUDE_gt_test_PUBLIC.json --lineage data/dude_metadata/licenses_metadata_datalineage.csv --output reports/replication_plan/dude_reproduced",
        "```", "",
        "The script uses only Python's standard library and the two archived v3 historical manifests. "
        "It rejects changed source hashes and nonempty output directories. The outputs are dude_metadata_feasibility.json/.md. "
        "The JSON includes per-doc counts and candidate question IDs/annotation rectangles for audit; those fields are not model input. "
        "No final panel, seed, native ROI or inference configuration is created.", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=PROJECT / "data/dude_metadata/2023-03-23_DUDE_gt_test_PUBLIC.json")
    parser.add_argument("--lineage", type=Path, default=PROJECT / "data/dude_metadata/licenses_metadata_datalineage.csv")
    parser.add_argument("--prior-main", type=Path, default=PROJECT / "reports/native100/provenance/main_manifest.jsonl")
    parser.add_argument("--prior-smoke", type=Path, default=PROJECT / "reports/native100/provenance/smoke_manifest.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = profile(args.metadata, args.lineage, args.prior_main, args.prior_smoke, args.output)
    print(json.dumps({"training": result["training"], "box_failures": result["boxes"]["first_failure_counts"],
                      "lineage_overlap_hits": len(result["lineage"]["historical_tatdqa_filename_overlap_hits"])}, indent=2))


if __name__ == "__main__":
    main()
