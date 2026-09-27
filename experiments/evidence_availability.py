"""V5 evidence-availability pilot with source-annotated privileged ROI geometry."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lookagain.runner import code_digest, sha256_file
from evidence_availability_core import ACTIONS, BUDGETS, FIDELITIES, FRESH_DESCRIPTION, ANSWER_INSTRUCTION, parse_action, validate_roi, build_request, score_response
from history_context import experiment_digest as v4_experiment_digest, validate_manifest as validate_v4_manifest
from tatdqa_metrics import score_official

BASE_SOURCE_SHA256 = "b897b6bf0e13c1045b1f1c77a732978ce91007ea87c513f2828547b3424003be"


def experiment_digest():
    digest = hashlib.sha256(v4_experiment_digest().encode())
    for name in ("evidence_availability.py", "evidence_availability_core.py", "prepare_evidence_roi.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_manifest(path):
    rows = validate_v4_manifest(path)
    root = Path(path).resolve().parent
    audit_path = root / "roi_audit.jsonl"
    audits = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit_map = {r["example_id"]: r for r in audits}
    if len(audit_map) != len(audits):
        raise ValueError("Duplicate ROI provenance entry")
    from PIL import Image
    for row in rows:
        with Image.open(root / row["image_path"]) as image:
            box = validate_roi(row["roi_pixels"], image.size)
            if (box[2]-box[0])*(box[3]-box[1]) > .25 * image.width * image.height:
                raise ValueError("ROI exceeds locked page-area fraction")
        audit = audit_map.get(row["example_id"])
        if audit is None or canonical_hash(audit) != row["roi_provenance_sha256"]:
            raise ValueError("ROI annotation provenance mismatch")
        if audit["roi_pixels"] != row["roi_pixels"]:
            raise ValueError("Manifest ROI differs from its annotation provenance")
    return rows


def validate_config(config):
    if list(config["actions"]) != list(ACTIONS) or config["overview_budgets"] != list(BUDGETS):
        raise ValueError("Exactly ten locked actions and three overview budgets required")
    expected = {"dtype": "bfloat16", "attention": "sdpa", "do_sample": False,
                "short_answer_prefix": "ANSWER:", "answer_max_tokens": 64,
                "crop_visual_tokens": 1024, "highres_visual_tokens": 4096}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Configuration violates frozen inference contract")
    if config["prompts"] != {"answer": ANSWER_INSTRUCTION, "fresh_context": FRESH_DESCRIPTION}:
        raise ValueError("Literal templates differ from frozen configuration")
    if config["protocol"]["inference_code_sha256"] != experiment_digest():
        raise ValueError("V5 inference/preparation code differs from lock")
    if v4_experiment_digest() != "c63cf16a860dc2b016454c183a63d031aaa98f55201100ab60c63b296d0d21f6":
        raise ValueError("Archived v4/v3 inference sources changed")
    protocol = Path(__file__).resolve().parents[1] / config["protocol"]["document"]
    if sha256_file(protocol) != config["protocol"]["document_sha256"]:
        raise ValueError("Protocol document changed")
    for key in ("rules_locked_at_utc", "main_execution_locked_at_utc"):
        if not config["protocol"][key]:
            raise ValueError("Final rules and code must be locked before inference")


def read_records(path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = row["example_id"], row["action"]
            if key in records or row["status"] != "ok":
                raise ValueError(f"Duplicate/failed record: {key}")
            records[key] = row
    return records


def validate_saved_identity(saved, identity, fingerprint):
    saved_identity = {key: saved.get(key) for key in identity}
    saved_digest = hashlib.sha256(json.dumps(saved_identity, sort_keys=True).encode()).hexdigest()
    if saved_identity != identity or saved.get("fingerprint") != fingerprint or saved_digest != fingerprint:
        raise ValueError("Run identity changed; use a new output directory")


def check_pair(records, example_id, budget):
    native = records.get((example_id, f"native_{budget}"))
    degraded = records.get((example_id, f"degraded_{budget}"))
    if native and degraded:
        for key in ("messages_sha256", "overview_size", "additional_size", "source_roi_pixels",
                    "projected_overview_roi_pixels", "image_grid_thw", "input_tokens", "visual_tokens"):
            if native[key] != degraded[key]:
                raise ValueError(f"Native/degraded {key} differs: {example_id}/{budget}")
        if native["image_rgb_sha256"][0] != degraded["image_rgb_sha256"][0]:
            raise ValueError("Paired overviews differ")
    available = [records[(example_id, f"native_{b}")] for b in BUDGETS if (example_id, f"native_{b}") in records]
    if available and any(r["image_rgb_sha256"][1] != available[0]["image_rgb_sha256"][1] for r in available):
        raise ValueError("Native ROI pixels vary across overview budgets")
    direct = records.get((example_id, f"direct_{budget}"))
    if direct:
        for regional in (native, degraded):
            if regional and direct["image_rgb_sha256"][0] != regional["image_rgb_sha256"][0]:
                raise ValueError("Regional overview differs from its matched direct image")


def validate_saved_records(records, rows, manifest, config):
    from PIL import Image, ImageOps
    for row in rows:
        present = [records[(row["example_id"], action)] for action in ACTIONS
                   if (row["example_id"], action) in records]
        if not present:
            continue
        with Image.open(manifest.parent / row["image_path"]) as raw:
            source = ImageOps.exif_transpose(raw).convert("RGB")
        for record in present:
            expected = {key: row[key] for key in ("example_id", "image_id", "source_id", "question")}
            expected.update(target_answer=row["answer"], source_image_sha256=row["image_sha256"],
                            roi_provenance_sha256=row["roi_provenance_sha256"])
            _, images, geometry = build_request(source, row["question"], record["action"], config, row["roi_pixels"])
            expected.update(geometry)
            scores = score_response(record["response"], row["answer"])
            expected.update(scores)
            expected.update(score_official(scores["predicted_answer"] or "", row["answer"]))
            expected["image_grid_thw"] = [[1, image.height // 16, image.width // 16] for image in images]
            expected["visual_tokens"] = sum(t*h*w//4 for t,h,w in expected["image_grid_thw"])
            if any(record.get(key) != value for key,value in expected.items()):
                raise ValueError(f"Saved record mismatch: {row['example_id']}/{record['action']}")
            if not record["answer_prefix_prefilled"] or record["response"] != "ANSWER:" + record["raw_continuation"]:
                raise ValueError("Saved answer continuation/prefix mismatch")
            if not 1 <= record["generated_tokens"] <= config["answer_max_tokens"]:
                raise ValueError("Invalid saved generation length")
            if record["elapsed_s"] <= 0 or record["input_tokens"] <= record["visual_tokens"]:
                raise ValueError("Invalid saved measurement")
        for budget in BUDGETS:
            check_pair(records, row["example_id"], budget)


def run(manifest, model_dir, config_path, output, smoke=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from PIL import Image, ImageOps
    from lookagain.backend import QwenBackend

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if code_digest() != BASE_SOURCE_SHA256:
        raise ValueError("Archived v1 inference sources changed")
    rows = validate_manifest(manifest)
    pins = config["protocol"]
    role = "smoke" if smoke else "main"
    if sha256_file(manifest) != pins[role + "_manifest_sha256"] or len(rows) != pins[role + "_examples"]:
        raise ValueError("Manifest hash/count violates protocol")
    if sha256_file(manifest.parent / "selection_metadata.json") != pins["preparation_metadata_sha256"]:
        raise ValueError("Prepared ROI metadata differs from lock")
    if sha256_file(manifest.parent / "roi_audit.jsonl") != pins["roi_audit_sha256"]:
        raise ValueError("ROI annotation audit differs from lock")
    model_meta = json.loads((model_dir / "lookagain-model.json").read_text(encoding="utf-8"))
    if model_meta["model_id"] != config["model_id"] or model_meta["revision"] != pins["model_revision"]:
        raise ValueError("Model identity mismatch")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    packages = {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "transformers", "Pillow", "numpy", "huggingface-hub")}
    runtime = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "packages": packages,
               "python": platform.python_version(), "platform": platform.platform()}
    identity = {"config": config, "manifest_sha256": sha256_file(manifest), "code_sha256": experiment_digest(),
                "model": model_meta, "runtime": runtime, "role": role, "example_ids": [r["example_id"] for r in rows]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output / "run.json"
    record_path = output / "records.jsonl"
    completion_path = output / "completed.json"
    if not meta_path.exists() and (record_path.exists() or completion_path.exists()):
        raise ValueError("Orphan records/completion marker without run identity")
    if meta_path.exists():
        validate_saved_identity(json.loads(meta_path.read_text(encoding="utf-8")), identity, fingerprint)
    else:
        metadata = {**identity, **runtime, "fingerprint": fingerprint, "base_code_sha256": code_digest(),
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "experiment": "evidence_availability_v5", "development_only": True, "label_privileged_localizer": True,
                    "cost_definition": "Synchronized full path including PNG decode, view construction, RGB hashes, processor, transfers, generation, decoding and logprob extraction. PDF rendering, model load, warmup and logging excluded; rendering recorded separately. Primary cost is standalone invocation; diagnostic decision-state cost adds the matched-budget direct call to a regional invocation. Localization is supplied by reference annotations and its cost is not measured. No cross-call KV cache reuse.",
                    "warmup": "Six synthetic calls: direct_256/native_256 at landscape, square and portrait shapes with a 256-pixel square ROI; two output tokens each. Nonexhaustive coverage.",
                    "durability": "Flush each action, fsync each document; no automatic retries; strict identity resume."}
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    completed = read_records(record_path)
    expected = {(r["example_id"], action) for r in rows for action in ACTIONS}
    if set(completed) - expected:
        raise ValueError("Unexpected record")
    validate_saved_records(completed, rows, manifest, config)
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if set(completed) != expected or completion["records_sha256"] != sha256_file(record_path) or completion["records"] != len(completed):
            raise ValueError("Completion marker does not match saved records")
    if expected <= completed.keys():
        if not (output / "completed.json").exists():
            (output / "completed.json").write_text(json.dumps({"records": len(completed), "elapsed_s": None,
                "note": "Completion marker recovered after all records were durable; total wall time unavailable.",
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "records_sha256": sha256_file(record_path)}, indent=2), encoding="utf-8")
        print("All declared actions already complete.", flush=True)
        return
    backend = QwenBackend(model_dir, config)
    for size in ((2400, 1800), (2000, 2000), (1800, 2400)):
        source = Image.new("RGB", size, (128, 128, 128))
        for action in ("direct_256", "native_256"):
            messages, images, _ = build_request(source, "What color is the image?", action, config, [128, 128, 384, 384])
            backend.generate(messages, images, 2, answer_prefix=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with record_path.open("a", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            alternatives = list(ACTIONS)
            item_seed = int(hashlib.sha256(f'{config["seed"]}:{row["example_id"]}'.encode()).hexdigest()[:16], 16)
            random.Random(item_seed).shuffle(alternatives)
            for action in alternatives:
                key = row["example_id"], action
                if key in completed:
                    continue
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    action_started = time.perf_counter()
                    with Image.open(manifest.parent / row["image_path"]) as raw:
                        source = ImageOps.exif_transpose(raw).convert("RGB")
                    messages, images, geometry = build_request(source, row["question"], action, config, row["roi_pixels"])
                    result = backend.generate(messages, images, config["answer_max_tokens"], answer_prefix=True)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - action_started
                    expected_grids = [[1, img.height // 16, img.width // 16] for img in images]
                    if result["image_grid_thw"] != expected_grids:
                        raise ValueError("Processor unexpectedly changed image grid")
                    scores = score_response(result["response"], row["answer"])
                    scores.update(score_official(scores["predicted_answer"] or "", row["answer"]))
                    record = {"example_id": row["example_id"], "image_id": row["image_id"], "source_id": row["source_id"],
                              "question": row["question"], "target_answer": row["answer"], "source_image_sha256": row["image_sha256"],
                              "action": action, "status": "ok", "roi_provenance_sha256": row["roi_provenance_sha256"], **result, **geometry, **scores, "elapsed_s": elapsed,
                              "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
                    completed[key] = record
                    for budget in BUDGETS:
                        check_pair(completed, row["example_id"], budget)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                except Exception as error:
                    with (output / "errors.jsonl").open("a", encoding="utf-8") as errors:
                        errors.write(json.dumps({"example_id": row["example_id"], "action": action, "error": repr(error)}) + "\n")
                    raise
            os.fsync(stream.fileno())
            print(f"[{index+1}/{len(rows)}] {len(completed)}/{len(expected)} actions complete; {(time.perf_counter()-started)/60:.1f} min after warmup", flush=True)
    (output / "completed.json").write_text(json.dumps({"records": len(completed), "elapsed_s": time.perf_counter()-started,
        "elapsed_s_scope": "Current execution segment after warmup; on resume excludes earlier segments.",
        "sum_measured_action_elapsed_s": sum(record["elapsed_s"] for record in completed.values()),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "records_sha256": sha256_file(record_path)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.manifest.resolve(), args.model_dir.resolve(), args.config.resolve(), args.output.resolve(), args.smoke)
