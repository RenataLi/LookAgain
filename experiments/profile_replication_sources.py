"""Profile report-disjoint TAT-DQA replication feasibility without model outputs.

Reads the pinned official training JSON/ZIP, archived v3 source manifests and
frozen preparation helpers. Writes a census and exclusion audit, never a sample,
rendered image, prompt, model call or final selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import time
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
PINS = {
    "train_json": "3025b4ca9c3c87f5ecd2c504371e1ee92929ef1f31d8a3ac38bc3a5376582fab",
    "train_zip": "412dde804ec9d5888b888a9d7179f2157bc2679b92379f9804e6f2062e7237d9",
    "prior_main": "757c3a4d024ac726f701c8732424e3457bfb292d224f67e2de22201ac641adc5",
    "prior_smoke": "ac4f35d9edf4c6e6a7b70225668fc89172a618f3eaecaa661009269c2c109564",
    "question_helper": "83f223acec84d40b902ab31fa927bb0fd989e5822e5f10dcb095c4e641da9787",
    "roi_helper": "213dc44f9b4bd45c8efe48e5ed9e68919ac3555d61e14e734c1093cef15f050f",
}
GENERIC_TABLE = re.compile(
    r"^\s*what (?:does|do) (?:the|this|these) tables? (?:show|present|represent|contain)\b", re.I
)
BETWEEN_YEARS = re.compile(
    r"\bhow much\b.*\bbetween\s+(?:19|20)\d{2}\s+and\s+(?:19|20)\d{2}", re.I
)
LEGAL_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "company",
                  "ltd", "limited", "plc", "nv", "sa", "ag"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pin(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Frozen input differs from its SHA256 pin: {path.name}")
    return actual


def report_key(source: str) -> str:
    """Conservative filename identity; unknown names fail instead of collapsing."""
    if (not isinstance(source, str) or "/" in source or "\\" in source
            or not source.casefold().endswith(".pdf")):
        raise ValueError("Expected a plain original-report PDF filename")
    match = re.fullmatch(r"(.+?)[_-]((?:19|20)\d{2})", source[:-4].casefold())
    if match is None:
        raise ValueError(f"Report filename lacks a terminal year: {source}")
    company = re.sub(r"[^a-z0-9]+", " ", match[1]).strip()
    if not company:
        raise ValueError("Empty normalized company name")
    return company + "|" + match[2]


def company_key(source: str) -> str:
    """Secondary alias screen; never used to merge different reports silently."""
    tokens = report_key(source).split("|")[0].split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def remaining_sources(sources: set[str], prior_sources: set[str]) -> set[str]:
    if not prior_sources <= sources:
        raise ValueError("Historical report source is missing from the raw release")
    prior_keys = {report_key(source) for source in prior_sources}
    return {source for source in sources if report_key(source) not in prior_keys}


def semantic_flags(question: str) -> list[str]:
    flags = []
    if GENERIC_TABLE.search(question):
        flags.append("generic_table_referent")
    if BETWEEN_YEARS.search(question):
        flags.append("between_years_level_vs_change_review")
    return flags


def validate_historical(rows: list[dict], entries: list[dict]) -> None:
    if len(rows) != 102 or len({row["source_id"] for row in rows}) != 102:
        raise ValueError("Historical exclusion must contain 102 distinct reports")
    if len({row["question_uid"] for row in rows}) != 102:
        raise ValueError("Duplicate historical question")
    lookup = {(entry["doc"]["uid"], q["uid"]): (entry["doc"], q)
              for entry in entries for q in entry["questions"]}
    for row in rows:
        doc, question = lookup.get((row["doc_uid"], row["question_uid"]), ({}, {}))
        if (doc.get("source") != row["source_id"]
                or question.get("question") != row["question"]
                or question.get("answer") != [row["answer"]]):
            raise ValueError("Historical manifest disagrees with raw annotation")


def validate_census(result: dict, questions: list[dict]) -> None:
    remaining = result["remaining"]
    profiles = remaining["source_profiles"]
    source_ids = {row["source_id"] for row in profiles}
    if len(source_ids) != len(profiles):
        raise ValueError("Duplicate source profile")
    if source_ids & set(result["prior_exclusion"]["source_ids"]):
        raise ValueError("Historical report leaked into remaining-source census")
    if len({row["question_uid"] for row in questions}) != len(questions):
        raise ValueError("Duplicate question UID in census")
    if any(row["source_id"] not in source_ids for row in questions):
        raise ValueError("Question has no source profile")
    eligible = [row for row in questions if row["eligible"]]
    filtered = [row for row in eligible
                if "generic_table_referent" not in row["semantic_review_flags"]]
    expected = {
        "source_upper_bound_before_filters": len(profiles),
        "remaining_questions": len(questions),
        "strict_eligible_questions": len(eligible),
        "strict_eligible_reports": len({row["source_id"] for row in eligible}),
        "strict_eligible_questions_if_generic_table_questions_removed": len(filtered),
        "strict_eligible_reports_if_generic_table_questions_removed": len({row["source_id"] for row in filtered}),
    }
    for key, value in expected.items():
        if remaining[key] != value:
            raise ValueError(f"Census count disagrees: {key}")
    if sum(remaining["question_first_failure_counts"].values()) + len(eligible) != len(questions):
        raise ValueError("Eligibility/exclusion walk does not reconcile")
    if sum(row["questions"] for row in profiles) != len(questions):
        raise ValueError("Per-report question totals do not reconcile")
    if sum(row["strict_eligible_questions"] for row in profiles) != len(eligible):
        raise ValueError("Per-report candidate totals do not reconcile")


def _load_helper(name: str, path: Path):
    # Called only after verifying helper hashes, and only for a full profile.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bound_inputs(raw: Path, prior_main: Path, prior_smoke: Path):
    paths = {
        "train_json": raw / "tat_dqa_dataset_train.json",
        "train_zip": raw / "tat_dqa_train.zip",
        "prior_main": prior_main,
        "prior_smoke": prior_smoke,
        "question_helper": PROJECT / "experiments/prepare_tatdqa.py",
        "roi_helper": PROJECT / "experiments/prepare_evidence_roi.py",
    }
    hashes = {name: verify_pin(path, PINS[name]) for name, path in paths.items()}
    entries = json.loads(paths["train_json"].read_text(encoding="utf-8"))
    prior = []
    for name, count in (("prior_main", 100), ("prior_smoke", 2)):
        rows = [json.loads(line) for line in paths[name].read_text(encoding="utf-8").splitlines()]
        if len(rows) != count:
            raise ValueError(f"Unexpected historical cohort size: {name}")
        prior.extend(rows)
    validate_historical(prior, entries)
    return paths, hashes, entries, prior


def _pdf_info(archive, uid: str, source: str, PdfReader) -> dict:
    blob = archive.read(f"train/{uid}.pdf")
    reader = PdfReader(io.BytesIO(blob))
    failure, geometry, dimensions, text_length = None, None, None, None
    if len(reader.pages) != 1:
        failure = "non_single_page_pdf"
    else:
        page = reader.pages[0]
        geometry = {
            "page_count": 1, "mediabox": list(map(float, page.mediabox)),
            "cropbox": list(map(float, page.cropbox)), "rotation": page.rotation,
            "user_unit": float(page.get("/UserUnit", 1)),
        }
        text_length = len((page.extract_text() or "").strip())
        media = geometry["mediabox"]
        dimensions = [math.ceil((media[2] - media[0]) * geometry["user_unit"] * 200 / 72),
                      math.ceil((media[3] - media[1]) * geometry["user_unit"] * 200 / 72)]
        if text_length < 100:
            failure = "pdf_extracted_text_under_100_chars"
        elif min(dimensions) < 32 or math.prod(dimensions) < 2 * 1024 * 32 * 32:
            failure = "predicted_200dpi_dimensions_below_threshold"
    return {
        "source_id": source, "source_pdf_sha256": hashlib.sha256(blob).hexdigest(),
        "failure": failure, "geometry": geometry, "predicted_render_size": dimensions,
        "extracted_text_chars": text_length,
    }


def profile(raw: Path, prior_main: Path, prior_smoke: Path, output: Path) -> dict:
    started = time.perf_counter()
    raw, prior_main, prior_smoke, output = [Path(p).resolve() for p in
                                           (raw, prior_main, prior_smoke, output)]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be absent or empty; existing reports are never overwritten")
    paths, hashes, entries, historical = _load_bound_inputs(raw, prior_main, prior_smoke)
    from pypdf import PdfReader
    question_helper = _load_helper("replication_question_helper", paths["question_helper"])
    roi_helper = _load_helper("replication_roi_helper", paths["roi_helper"])
    old_sources = {row["source_id"] for row in historical}
    sources = {entry["doc"]["source"] for entry in entries}
    pool = remaining_sources(sources, old_sources)
    canonical_groups, company_groups = defaultdict(list), defaultdict(list)
    uid_sources, qid_sources = defaultdict(set), defaultdict(set)
    for source in sorted(sources):
        canonical_groups[report_key(source)].append(source)
        company_groups[company_key(source)].append(source)
    for entry in entries:
        uid_sources[entry["doc"]["uid"]].add(entry["doc"]["source"])
        for question in entry["questions"]:
            qid_sources[question["uid"]].add(entry["doc"]["source"])
    by_source = {source: {
        "source_id": source, "report_key": report_key(source), "company_key": company_key(source),
        "documents": 0, "questions": 0, "metadata_eligible_questions": 0,
        "strict_eligible_questions": 0, "eligible_doc_uids": set(), "failures": Counter(),
    } for source in pool}
    counts, failures, pdf_failures, geometry_counts = Counter(), Counter(), Counter(), Counter()
    flag_counts, ocr_page_counts = Counter(), Counter()
    metadata_sources, questions, pdf_cache = set(), [], {}
    pdf_hash_sources = defaultdict(set)
    with zipfile.ZipFile(paths["train_zip"]) as archive:
        members = archive.infolist()
        if len(members) != len({member.filename for member in members}):
            raise ValueError("Duplicate archive members")
        member_types = Counter(Path(member.filename).suffix.lower() for member in members
                               if not member.is_dir())
        archive_sizes = {
            suffix: sum(member.file_size for member in members
                        if Path(member.filename).suffix.lower() == suffix)
            for suffix in member_types
        }
        for entry in entries:
            doc, source = entry["doc"], entry["doc"]["source"]
            uid = doc["uid"]
            if not re.fullmatch("[a-fA-F0-9]{32}", uid):
                raise ValueError("Unsafe document UID")
            ocr = json.loads(archive.read(f"train/{uid}.json"))
            pages = ocr.get("pages")
            ocr_page_counts[len(pages) if isinstance(pages, list) else "missing"] += 1
            if source in pool:
                by_source[source]["documents"] += 1
                by_source[source]["questions"] += len(entry["questions"])
            for question in entry["questions"]:
                failure = ("non_single_page_ocr" if not isinstance(pages, list) or len(pages) != 1
                           else question_helper.question_failure(question))
                if failure is None:
                    metadata_sources.add(source)
                if source not in pool:
                    continue
                counts["remaining_questions"] += 1
                row = {
                    "source_id": source, "doc_uid": uid, "question_uid": question["uid"],
                    "question": question["question"], "answer": question["answer"],
                    "eligible": False, "first_exclusion_reason": None,
                }
                if failure is None:
                    by_source[source]["metadata_eligible_questions"] += 1
                    counts["remaining_metadata_eligible_questions"] += 1
                    try:
                        evidence = roi_helper.evidence_words(question, ocr)
                        if roi_helper.normalize_span(evidence["selected_text"]) != roi_helper.normalize_span(question["answer"][0]):
                            raise roi_helper.Ineligible("selected_text_reference_mismatch")
                        if evidence["unsafe_boundary_cut"]:
                            raise roi_helper.Ineligible("alphanumeric_or_hyphen_boundary_cut")
                        if uid not in pdf_cache:
                            pdf = _pdf_info(archive, uid, source, PdfReader)
                            pdf_cache[uid] = pdf
                            pdf_hash_sources[pdf["source_pdf_sha256"]].add(source)
                            if pdf["failure"]:
                                pdf_failures[pdf["failure"]] += 1
                            if pdf["geometry"] is not None:
                                geometry = pdf["geometry"]
                                geometry_counts["unequal_crop_media"] += int(geometry["mediabox"] != geometry["cropbox"])
                                geometry_counts["nonzero_rotation"] += int(geometry["rotation"] != 0)
                                geometry_counts["nondefault_user_unit"] += int(geometry["user_unit"] != 1)
                        pdf = pdf_cache[uid]
                        if pdf["failure"]:
                            raise roi_helper.Ineligible(pdf["failure"])
                        roi, construction = roi_helper.padded_roi(
                            [word["bbox"] for word in evidence["selected_words"]],
                            evidence["ocr_page_bbox"], pdf["geometry"], pdf["predicted_render_size"],
                        )
                        flags = semantic_flags(question["question"])
                        flag_counts.update(flags)
                        row.update(
                            eligible=True, roi_pixels_predicted=roi,
                            roi_area_fraction=construction["roi_page_area_fraction"],
                            mapped_text=evidence["selected_text"], semantic_review_flags=flags,
                            source_pdf_sha256=pdf["source_pdf_sha256"],
                            predicted_render_size=pdf["predicted_render_size"],
                        )
                        by_source[source]["strict_eligible_questions"] += 1
                        by_source[source]["eligible_doc_uids"].add(uid)
                        counts["strict_eligible_questions"] += 1
                    except roi_helper.Ineligible as error:
                        failure = str(error)
                if failure:
                    row["first_exclusion_reason"] = failure
                    failures[failure] += 1
                    by_source[source]["failures"][failure] += 1
                questions.append(row)
        # Exact-byte check spans all report sources, including historical reports.
        for entry in entries:
            uid, source = entry["doc"]["uid"], entry["doc"]["source"]
            if uid not in pdf_cache:
                digest = hashlib.sha256(archive.read(f"train/{uid}.pdf")).hexdigest()
                pdf_hash_sources[digest].add(source)

    profiles = []
    for source in sorted(by_source):
        value = by_source[source]
        value["eligible_doc_uids"] = sorted(value["eligible_doc_uids"])
        value["failures"] = dict(value["failures"])
        profiles.append(value)
    eligible = [row for row in questions if row["eligible"]]
    generic_filtered = [row for row in eligible if "generic_table_referent" not in row["semantic_review_flags"]]
    both_filtered = [row for row in eligible if not row["semantic_review_flags"]]
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Source-only census and feasibility; no final sampling, model outcomes, inference or rendering.",
        "bindings": {
            "source_file_sha256": hashes,
            "profiler_sha256": sha256_file(Path(__file__)),
            "input_locations": {
                "train_json": "RAW/tat_dqa_dataset_train.json", "train_zip": "RAW/tat_dqa_train.zip",
                "prior_main": "reports/native100/provenance/main_manifest.jsonl",
                "prior_smoke": "reports/native100/provenance/smoke_manifest.jsonl",
                "question_helper": "experiments/prepare_tatdqa.py",
                "roi_helper": "experiments/prepare_evidence_roi.py",
            },
            "input_location_note": "RAW is --raw; archived manifest labels are defaults and CLI overrides must have identical pinned bytes.",
        },
        "release": {
            "document_entries": len(entries), "question_count": sum(len(e["questions"]) for e in entries),
            "original_report_source_count": len(sources),
            "report_years": dict(Counter(report_key(source).split("|")[-1] for source in sources)),
            "unique_doc_uids": len(uid_sources), "unique_question_uids": len(qid_sources),
            "ocr_page_counts": dict(ocr_page_counts), "archive_member_types": dict(member_types),
            "archive_uncompressed_bytes_by_type": archive_sizes,
            "metadata_eligible_reports_all": len(metadata_sources),
        },
        "prior_exclusion": {
            "unique_source_ids": len(old_sources), "source_ids": sorted(old_sources),
            "rule": "Exclude every fragment, page and question from all 102 historical v3 report sources; v5 adds no sources.",
        },
        "aliases": {
            "canonical_report_collisions": {key: value for key, value in canonical_groups.items() if len(value) > 1},
            "legal_suffix_company_collisions": {key: value for key, value in company_groups.items() if len(value) > 1},
            "doc_uid_cross_source_collisions": {key: sorted(value) for key, value in uid_sources.items() if len(value) > 1},
            "question_uid_cross_source_collisions": {key: sorted(value) for key, value in qid_sources.items() if len(value) > 1},
            "exact_pdf_bytes_cross_source_collisions": [sorted(value) for value in pdf_hash_sources.values() if len(value) > 1],
            "limitations": "Filename and exact-PDF checks cannot establish absence of corporate aliases, rebrands, subsidiaries or semantically duplicate pages.",
        },
        "remaining": {
            "source_upper_bound_before_filters": len(pool),
            "original_source_ids": sorted(pool), "metadata_eligible_reports": len(pool & metadata_sources),
            "strict_eligible_reports": len({row["source_id"] for row in eligible}),
            "strict_eligible_original_source_ids": sorted({row["source_id"] for row in eligible}),
            **dict(counts),
            "question_first_failure_counts": dict(failures),
            "pdf_documents_screened": len(pdf_cache), "pdf_first_failure_counts": dict(pdf_failures),
            "geometry_counts_among_screened_pdf_documents": dict(geometry_counts),
            "strict_eligible_questions_if_generic_table_questions_removed": len(generic_filtered),
            "strict_eligible_reports_if_generic_table_questions_removed": len({row["source_id"] for row in generic_filtered}),
            "strict_eligible_reports_if_both_review_flags_removed": len({row["source_id"] for row in both_filtered}),
            "semantic_review_flag_question_counts": dict(flag_counts), "source_profiles": profiles,
        },
        "proposed_semantic_screens": {
            "frozen_selection_rule": False,
            "generic_table_regex": GENERIC_TABLE.pattern, "between_years_regex": BETWEEN_YEARS.pattern,
            "note": "Sensitivity counts only. Exact mapping does not prove correct row, year, unit or question interpretation. Source-only full-page semantic review is still needed before a future inference panel.",
        },
        "rendering": {
            "rendered": False, "released224_pngs_used": False,
            "dpi": 200, "mode": "RGB", "page_box": "full MediaBox",
            "predicted_size_method": "ceil(MediaBox extent * UserUnit *200/72); recheck actual raster dimensions after future rendering.",
            "source_member_template": "train/{32hex_doc_uid}.pdf and train/{32hex_doc_uid}.json",
            "poppler_command": "pdftoppm -f 1 -singlefile -r 200 -png INPUT.pdf OUTPUT_PREFIX",
        },
        "published_context": {
            "primary_source": "https://arxiv.org/pdf/2207.11871.pdf",
            "location": "Section 3.3, page 3",
            "published_total_original_reports": 182,
            "conditional_report_ceiling_after102_historical": 80,
            "status": "Context from the published paper, not a locally computed train/dev/test source union.",
            "limitation": "The paper reports 2758 document fragments/3067 pages across splits; splitting is by fragment. Do not infer strict eligibility across unseen splits or add10 to the local62.",
        },
        "selection_frozen": False, "random_seed_selected": None, "model_outputs_used": False,
        "duration_seconds": time.perf_counter() - started,
    }
    validate_census(result, questions)
    output.mkdir(parents=True, exist_ok=True)
    walk = output / "source_question_walk.jsonl"
    walk.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions), encoding="utf-8", newline="\n")
    result["bindings"]["question_walk_sha256"] = sha256_file(walk)
    result["outputs"] = {
        "summary_json": "source_feasibility.json", "summary_markdown": "source_feasibility.md",
        "question_walk_jsonl": walk.name,
        "question_schema": {
            "all_rows": ["source_id", "doc_uid", "question_uid", "question", "answer", "eligible", "first_exclusion_reason"],
            "eligible_rows_add": ["roi_pixels_predicted", "roi_area_fraction", "mapped_text", "semantic_review_flags", "source_pdf_sha256", "predicted_render_size"],
        },
    }
    (output / "source_feasibility.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "source_feasibility.md").write_text(render_markdown(result), encoding="utf-8", newline="\n")
    return result


def render_markdown(result: dict) -> str:
    r = result["remaining"]
    lines = [
        "# Feasibility of an independent report panel", "",
        "This is a source-only census of the pinned official TAT-DQA training release. "
        "No model outputs, final selection, seed, inference or new page rendering were used.", "",
        "| Source-only step | Original reports | Candidate questions |",
        "| --- | ---: | ---: |",
        f"| Full local training release | {result['release']['original_report_source_count']} | {result['release']['question_count']} |",
        f"| Historical reports excluded completely | {result['prior_exclusion']['unique_source_ids']} | — |",
        f"| Remaining before eligibility filters | {r['source_upper_bound_before_filters']} | {r['remaining_questions']} |",
        f"| Single-page OCR and fixed question metadata rules | {r['metadata_eligible_reports']} | {r['remaining_metadata_eligible_questions']} |",
        f"| Strict mapping, PDF and predicted ROI geometry | {r['strict_eligible_reports']} | {r['strict_eligible_questions']} |",
        f"| Sensitivity: remove generic table-referent questions | {r['strict_eligible_reports_if_generic_table_questions_removed']} | {r['strict_eligible_questions_if_generic_table_questions_removed']} |", "",
        "**The local archive cannot supply 500–640 new independent original reports.** "
        "Multiple fragments, pages or questions from one report do not become independent reports. "
        "The 62-report figure is a technical feasibility ceiling before new semantic review, not a finalized sample.", "",
        "## Rules and source quality", "",
        "Eligibility reuses the exact pinned v3 question filter and v5 annotation/geometry helpers: "
        "one actual PDF/OCR page; one nonempty span answer; empty scale and no comparison; "
        "answer≤160 characters and 25 whitespace words; extractable PDF text≥100 characters; "
        "predicted full 200 DPI raster area≥2,097,152 pixels; exact casefold/whitespace mapped-reference equality; "
        "aligned whole-word geometry without alphanumeric/hyphen cuts; rotation 0, UserUnit 1, contained CropBox; "
        "reconciled OCR aspect; two median word-height margins, minimum 256-square ROI and area≤25% of the page.", "",
        "The old manual quality mask is not applied to new questions. "
        "The 36 generic-table exclusions are a proposed lexical sensitivity analysis, not a frozen rule. "
        "A source-only full-page review must still check question referents, rows, dates, units and reference correctness. "
        "Exact mapped-text equality cannot establish those facts.", "",
        "## Report identity", "",
        "All local filenames are 2019 reports. Entire canonical company-year report sources are excluded, "
        "including every old source's other pages/questions. Filename case/punctuation and legal-suffix screens, "
        "document/question UID checks and cross-source exact-PDF-byte checks found no collisions. "
        "These checks do not prove absence of corporate aliases or semantically duplicate content.", "",
        "## Published scope versus local computation", "",
        "The original [TAT-DQA paper, §3.3, p.3](https://arxiv.org/pdf/2207.11871.pdf) reports 182 financial reports "
        "in total, represented by 2758 document fragments/3067 pages. Its splits are at the fragment level. "
        "The arithmetic 182−102 gives a conditional ceiling of 80 previously unused reports under that published inventory; "
        "it is not a computed union of locally downloaded splits. No validation/test assets were downloaded for this profile. "
        "Do not add 10 to 62 as a strict eligible total: other splits may add questions to existing report IDs, "
        "and their mapping/geometry/semantics have not been checked.", "",
        "## Reproduction", "",
        "Run from the repository root with Python and the project's report dependencies installed "
        "(Pillow and pypdf are needed; no model or GPU is loaded). Place the pinned official training files under "
        "data/tatdqa_raw, or pass another directory using --raw. The archived historical manifests are included in "
        "reports/native100/provenance.", "",
        "```text",
        "python experiments/profile_replication_sources.py --raw data/tatdqa_raw --output reports/replication_plan/reproduced",
        "```", "",
        "The output directory must be absent or empty. Optional --prior-main/--prior-smoke paths must contain "
        "the exact pinned archived bytes. The script verifies raw-release, historical-manifest and helper hashes before "
        "profiling. It writes source_feasibility.json, source_feasibility.md and source_question_walk.jsonl, "
        "with the question-walk hash bound in the JSON. It neither samples questions nor chooses a seed.", "",
        "For a future selected panel, render the original ZIP PDF member at 200 DPI using full MediaBox, RGB "
        "and the documented Poppler command; do not use released 224px thumbnails. Current dimensions/ROI boxes "
        "are predictions from PDF geometry and require actual-raster verification before inference.", "",
        "## Output schemas", "",
        "The JSON contains bindings, release, prior_exclusion, aliases, remaining, proposed_semantic_screens, "
        "rendering and published_context. remaining.source_profiles and original_source_ids cover every remaining "
        "original report. The JSONL has one row per remaining question: source/document/question identity, original "
        "question and answer, eligibility and first failure; passing rows add mapped text, predicted ROI/size, "
        "PDF hash and semantic-screen flags. This annotation audit is never prompt content.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=PROJECT / "data/tatdqa_raw",
                        help="Directory containing the two pinned official training assets")
    parser.add_argument("--prior-main", type=Path, default=PROJECT / "reports/native100/provenance/main_manifest.jsonl")
    parser.add_argument("--prior-smoke", type=Path, default=PROJECT / "reports/native100/provenance/smoke_manifest.jsonl")
    parser.add_argument("--output", type=Path, required=True,
                        help="Absent or empty output directory; existing files are never overwritten")
    args = parser.parse_args()
    result = profile(args.raw, args.prior_main, args.prior_smoke, args.output)
    keys = ("source_upper_bound_before_filters", "metadata_eligible_reports", "strict_eligible_reports",
            "strict_eligible_questions", "strict_eligible_questions_if_generic_table_questions_removed")
    print(json.dumps({"original_reports": result["release"]["original_report_source_count"],
                      "historical_reports": result["prior_exclusion"]["unique_source_ids"],
                      **{key: result["remaining"][key] for key in keys}}, indent=2))


if __name__ == "__main__":
    main()

