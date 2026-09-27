"""Reserve source-disjoint region-selection splits without model generation.

Reads historical records only to exclude source identities, never to rank cases.
Labels and five-field model observations are separate artifacts. Original images
are copied byte-for-byte; no source media is copied to the public output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

SEED = 20260927
ADAPTER_SHA = "050ed758985c82c89062e154b6d3869acd53d35695dc4c30407473ee7dbbec3b"
SOURCE_LOCK_SHA = "c684b0b59af32e68dd30d72e04b3d61b1d28f16e78d250d8a983021316cb86fd"
OBSERVATION_FIELDS = ("example_id", "question", "image_path", "image_sha256", "source_cluster_id")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def index(rows):
    result = {x["source_cluster_id"]: x for x in rows}
    require(len(result) == len(rows), "Duplicate source cluster")
    require(len({x["example_id"] for x in rows}) == len(rows), "Duplicate example ID")
    return result


def ranked(rows, cohort):
    return sorted(rows, key=lambda x: hashlib.sha256(
        f"region-{cohort}:{SEED}:{x['source_cluster_id']}".encode()).hexdigest())


def observation(row):
    return {field: row[field] for field in OBSERVATION_FIELDS}


def safe(root, relative):
    path = (Path(root) / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(Path(root).resolve()), "Escaping image path")
    return path


def write_lines(path, rows):
    Path(path).write_text("".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in rows), encoding="utf-8")


def prepare(args):
    require(not args.output.exists(), "Data output already exists; no overwrite")
    require(not args.public_output.exists(), "Public output already exists; no overwrite")
    adapter_path = Path(__file__).with_name("prepare_dude_replication.py")
    require(sha(adapter_path) == ADAPTER_SHA, "Frozen source adapter changed")
    require(sha(args.source_lock) == SOURCE_LOCK_SHA, "Frozen source-design lock changed")
    spec = importlib.util.spec_from_file_location("_frozen_dude_source_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    lock, census = read(args.source_lock), read(args.census)
    selection = read(args.final_data / "selection_metadata.json")
    main_path, eng_path = args.final_data / "main_manifest.jsonl", args.final_data / "engineering_manifest.jsonl"
    for path, key in ((main_path, "main_manifest_sha256"), (eng_path, "engineering_manifest_sha256")):
        require(sha(path) == selection[key] == lock["public_file_bindings"][f"reports/dude_preparation/{path.name}"], "Prior final manifest changed")
    require(sha(args.census) == selection["census_sha256"], "Census binding differs")
    require(sha(args.reviews) == selection["review_sha256"], "Combined source-review ledger changed")
    main, engineering = index(lines(main_path)), index(lines(eng_path))
    candidates, reviews = index(census["candidates"]), index(lines(args.reviews))
    reservations = set(selection["engineering_source_reservations"])
    require(len(main) == 660 and len(engineering) == 10 and len(reservations) == 11, "Prior cohort counts differ")
    require(reservations == set(census["engineering_reserved_source_clusters"]), "Engineering exclusions differ")
    packs, rendered, failures = {}, {}, {}
    for pack in sorted(args.audit_packs):
        prep = read(pack / "preparation.json")
        require(prep["census_sha256"] == sha(args.census) and prep["adapter_sha256"] == ADAPTER_SHA, "Audit pack source binding differs")
        require(sha(pack / "candidate_manifest.jsonl") == prep["candidate_manifest_sha256"], "Candidate manifest changed")
        require(sha(pack / "source_audit.jsonl") == prep["source_audit_sha256"], "Source audit changed")
        require(sha(pack / "preparation.json") == selection["audit_pack_sha256"][pack.name], "Unbound audit pack")
        audits = index(lines(pack / "source_audit.jsonl"))
        rows = index(lines(pack / "candidate_manifest.jsonl"))
        require(set(audits) == set(rows), "Pack audit coverage differs")
        for cid, row in rows.items():
            require(cid not in rendered, "Overlapping packs")
            require(canonical(audits[cid]) == row["roi_provenance_sha256"], "Source audit row binding differs")
            adapter.validate_candidate_binding(row, audits[cid], candidates[cid])
            rendered[cid] = (row, safe(pack, row["image_path"]), pack / "candidate_manifest.jsonl")
        for failure in prep["failures"]:
            cid = failure["source_cluster_id"]
            adapter.validate_render_failure(failure, candidates[cid])
            require(cid not in failures, "Duplicate render failure")
            failures[cid] = failure
        packs[pack.name] = sha(pack / "preparation.json")
    require(set(packs) == set(selection["audit_pack_sha256"]), "Incomplete prior audit pack inventory")
    for cid, review in reviews.items():
        adapter.validate_source_review(rendered[cid][0], review)
    # Records are inspected solely for identity exclusion, never performance.
    historical, run_inventory = set(), []
    for path in sorted(args.runs.glob("*/records.jsonl")):
        rows = lines(path)
        found = set()
        for row in rows:
            cid = row.get("source_cluster_id")
            if isinstance(cid, str) and cid.startswith("dude-source:"):
                found.add(cid)
            elif str(row.get("example_id", "")).startswith("dude:"):
                matches = [c for c, value in candidates.items() if value["example_id"] == row["example_id"]]
                require(len(matches) == 1, "Unresolved DUDE outcome identity")
                found.add(matches[0])
        historical.update(found)
        run_inventory.append({"run_name": path.parent.name, "records_sha256": sha(path), "record_count": len(rows), "dude_source_count": len(found)})
    require(set(main) | set(engineering) <= historical, "Existing outcomes missing from run inventory")
    extras = [rendered[cid][0] for cid, review in reviews.items()
              if review["semantic_approved"] and cid not in set(main) | reservations | historical]
    require(len(extras) == 60, "Expected all60 approved outcome-unseen extras; do not silently lower N")
    chosen = {"development": ranked(list(main.values()), "development")[:64],
              "evaluation": ranked(extras, "evaluation"),
              "engineering": ranked(list(engineering.values()), "engineering")[:2]}
    selected = [row for rows in chosen.values() for row in rows]
    require(len(index(selected)) == 126, "New cohorts overlap")
    evaluation_ids = {x["source_cluster_id"] for x in chosen["evaluation"]}
    require(not evaluation_ids & (historical | reservations | set(main)), "Evaluation contamination")
    require(not any(row["near_duplicate_review_flags"] for row in selected), "Unresolved near-duplicate flag in selected sources")
    require(not {x["source_pdf_sha256"] for x in chosen["evaluation"]} & {x["source_pdf_sha256"] for x in list(main.values()) + list(engineering.values())}, "Exact PDF overlap")
    # Validate all selected media and final-row promotion before writing anything.
    prepared = {}
    for cohort, rows in chosen.items():
        prepared[cohort] = []
        for row in rows:
            cid = row["source_cluster_id"]
            base, image, upstream = rendered[cid]
            review = reviews[cid]
            promoted = {**base, "validated_primary_answers": review["validated_primary_answers"],
                        "source_semantic_audit_status": "approved", "source_semantic_review_sha256": canonical(review),
                        "cohort": "engineering" if cid in reservations else "main"}
            if cohort != "evaluation":
                require(promoted == row, "Existing final row differs from source/review reconstruction")
                image, upstream = safe(args.final_data, row["image_path"]), main_path if cohort == "development" else eng_path
            else:
                promoted["cohort"] = base["cohort"]  # Preserve the original candidate cohort metadata.
            require(sha(image) == promoted["image_sha256"], "Selected source image changed")
            updated = {**promoted, "image_path": "images/" + hashlib.sha256(row["example_id"].encode()).hexdigest() + ".png",
                       "region_selection_cohort": cohort, "upstream_image_path": row["image_path"],
                       "upstream_row_sha256": canonical(row), "upstream_manifest_sha256": sha(upstream)}
            require(updated["question"] == row["question"] and updated["original_answers"] == row["original_answers"], "Question/reference mutation")
            prepared[cohort].append((updated, image))
    pool = set(candidates) - set(main) - reservations - historical
    categories = Counter()
    untouched = []
    for cid in pool:
        category = (("source_approved" if reviews[cid]["semantic_approved"] else "source_rejected") if cid in reviews
                    else "render_failed" if cid in failures else "rendered_unreviewed" if cid in rendered else "never_rendered_or_visually_reviewed")
        categories[category] += 1
        if category == "never_rendered_or_visually_reviewed":
            untouched.append({"source_cluster_id": cid, "example_id": candidates[cid]["example_id"],
                              "source_rank": candidates[cid]["source_rank"], "near_duplicate_review_flags": candidates[cid]["near_duplicate_review_flags"]})
    require(len(untouched) == 443, "Untouched future-pool inventory changed")
    args.output.mkdir(parents=True)
    args.public_output.mkdir(parents=True)
    cohort_metadata = {}
    for cohort, entries in prepared.items():
        target = args.output / cohort
        (target / "images").mkdir(parents=True)
        labels = [row for row, _ in entries]
        observations = [observation(row) for row in labels]
        for row, image in entries:
            shutil.copyfile(image, target / row["image_path"])
            require(sha(target / row["image_path"]) == row["image_sha256"], "Copied media differs")
        for name, rows in (("labels.jsonl", labels), ("observation_manifest.jsonl", observations)):
            write_lines(target / name, rows)
            shutil.copyfile(target / name, args.public_output / f"{cohort}_{name}")
        cohort_metadata[cohort] = {"n": len(labels), "source_cluster_ids": [r["source_cluster_id"] for r in labels],
                                   "example_ids": [r["example_id"] for r in labels],
                                   "labels_sha256": sha(target / "labels.jsonl"),
                                   "observation_manifest_sha256": sha(target / "observation_manifest.jsonl")}
    metadata = {"schema_version": 1, "stage": "region_selection_splits_reserved_before_new_generation",
                "reserved_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "new_vlm_calls": 0,
                "selection_used_model_scores": False, "execution_locked": False, "cohorts": cohort_metadata,
                "observation_fields": list(OBSERVATION_FIELDS), "answer_page_privileged": True,
                "gold_roi_available_in_observations": False, "evaluation_source_only_audit_previously_visible": True,
                "evaluation_model_outcomes_previously_generated": False,
                "evaluation_rule": "All60 source-approved extras; no new evaluation source, label, prompt or threshold tuning after reservation.",
                "development_rule": "First64 prior main sources by SHA256('region-development:20260927:'+source_cluster_id).",
                "engineering_rule": "First2 prior usable engineering sources by SHA256('region-engineering:20260927:'+source_cluster_id).",
                "evaluation_order": "SHA256('region-evaluation:20260927:'+source_cluster_id)",
                "source_pool_counts": dict(categories), "historical_outcome_source_ids": sorted(historical),
                "engineering_reserved_source_ids": sorted(reservations), "historical_runs": run_inventory,
                "future_pool": {"n": len(untouched), "status": "reserved_generically_not_eligible_or_visually_audited",
                                "sources": sorted(untouched, key=lambda x: x["source_cluster_id"])},
                "bindings": {"script_sha256": sha(__file__), "source_adapter_sha256": ADAPTER_SHA,
                             "source_design_lock_sha256": sha(args.source_lock), "census_sha256": sha(args.census),
                             "combined_source_reviews_sha256": sha(args.reviews), "selection_metadata_sha256": sha(args.final_data / "selection_metadata.json"),
                             "audit_pack_sha256": packs},
                "limitations": ["Evaluation is model-outcome-held-out, not unseen to source preparation or independent human annotation.",
                                "The original answer page remains privileged; the selector receives only pixels, question and opaque identities/path.",
                                "Label separation is an interface contract, not operating-system access control. Never pass label manifests to a selector.",
                                "Known exact-source clustering and near-duplicate checks do not exclude all residual template or pretraining overlap.",
                                "The443 future candidates require source/render review; two have a documented near-duplicate flag.",
                                "Only recorded runs under the explicitly supplied runs directory are inventoried; earlier known source-only probe is excluded through engineering reservation.",
                                "N60 is a bounded pilot, not an80%-powered confirmation of a small effect."]}
    payload = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (args.output / "split_reservation.json").write_text(payload, encoding="utf-8")
    (args.public_output / "split_reservation.json").write_text(payload, encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("census", "reviews", "final-data", "source-lock", "runs", "output", "public-output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--audit-packs", type=Path, nargs="+", required=True)
    result = prepare(parser.parse_args())
    print(json.dumps({"status": "PASS", "counts": {k: v["n"] for k, v in result["cohorts"].items()}, "new_vlm_calls": 0}))


if __name__ == "__main__":
    main()
