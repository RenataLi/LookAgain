"""Frozen v4 history-context intervention, importing immutable v3 pixels/scoring."""
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
from history_context_core import ACTIONS, HISTORIES, FIDELITIES, REGIONS, PLACEHOLDER, FRESH_DESCRIPTION, ANSWER_INSTRUCTION, CROP_PROMPT, REPEAT_PROMPT, build_request, score_response
from native_detail import experiment_digest as v3_experiment_digest
from tatdqa_metrics import score_official

BASE_SOURCE_SHA256 = "b897b6bf0e13c1045b1f1c77a732978ce91007ea87c513f2828547b3424003be"


def experiment_digest():
    digest = hashlib.sha256(v3_experiment_digest().encode())
    for name in ("history_context.py", "history_context_core.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def validate_manifest(path):
    from PIL import Image
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty manifest")
    for field in ("example_id", "image_id", "source_id"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"Repeated {field}")
    for row in rows:
        for field in ("example_id", "image_id", "source_id", "question", "answer", "image_path", "image_sha256"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing/non-string {field}")
        relative = Path(row["image_path"])
        if relative.is_absolute() or PureWindowsPath(str(relative)).drive:
            raise ValueError("Image path must be relative")
        image_path = (path.parent / relative).resolve()
        if not image_path.is_relative_to(path.parent):
            raise ValueError("Image escapes manifest directory")
        if sha256_file(image_path) != row["image_sha256"]:
            raise ValueError("Rendered source image hash mismatch")
        with Image.open(image_path) as image:
            image.load()
            if min(image.size) < 32 or image.width * image.height < 2 * 1024 * 32 * 32:
                raise ValueError("Source below locked detail resolution criterion")
    return rows


def validate_config(config):
    if list(config["actions"]) != list(ACTIONS):
        raise ValueError("Exactly twenty-seven locked actions are required")
    expected = {"dtype": "bfloat16", "attention": "sdpa", "do_sample": False,
                "short_answer_prefix": "ANSWER:", "answer_max_tokens": 64,
                "base_visual_tokens": 1024, "crop_visual_tokens": 1024, "highres_visual_tokens": 4096}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Configuration violates frozen inference contract")
    expected_prompts = {"answer": ANSWER_INSTRUCTION, "crop": CROP_PROMPT, "repeat": REPEAT_PROMPT,
                        "fresh_context": FRESH_DESCRIPTION, "placeholder": PLACEHOLDER}
    if config["prompts"] != expected_prompts:
        raise ValueError("Literal templates differ from frozen configuration")
    if config["protocol"]["inference_code_sha256"] != experiment_digest():
        raise ValueError("V4 inference code differs from lock")
    if v3_experiment_digest() != "515dc6941bfe514a3beffba157d8b31a54ae061525447a1653c1dc57bcda57cd":
        raise ValueError("Archived v3 inference sources changed")
    protocol_path = Path(__file__).resolve().parents[1] / config["protocol"]["document"]
    if sha256_file(protocol_path) != config["protocol"]["document_sha256"]:
        raise ValueError("Protocol document changed")
    if not config["protocol"]["main_execution_locked_at_utc"]:
        raise ValueError("Execution lock is required before any inference")
    if not config["protocol"]["rules_locked_at_utc"]:
        raise ValueError("Protocol must be locked before inference")


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


def check_pair(records, example_id, region):
    for history in HISTORIES:
        left = records.get((example_id, f"{history}_native_{region}"))
        right = records.get((example_id, f"{history}_degraded_{region}"))
        if left and right:
            for key in ("messages_sha256", "followup_prompt", "overview_size", "additional_size", "image_grid_thw", "input_tokens", "visual_tokens"):
                if left[key] != right[key]:
                    raise ValueError(f"Mismatched native/degraded {key}: {example_id}/{history}/{region}")
            if left["image_rgb_sha256"][0] != right["image_rgb_sha256"][0]:
                raise ValueError("Paired first images differ")
    for fidelity in FIDELITIES:
        present = [records[(example_id, f"{history}_{fidelity}_{region}")]
                   for history in HISTORIES if (example_id, f"{history}_{fidelity}_{region}") in records]
        if present and any(record["image_rgb_sha256"] != present[0]["image_rgb_sha256"] for record in present):
            raise ValueError("Identical fidelity/region pixels differ across histories")


def validate_saved_records(records, rows, manifest, config):
    """A matching run.json cannot legitimize altered or orphan record contents."""
    from PIL import Image, ImageOps
    for row in rows:
        present = [records[(row["example_id"], action)] for action in ACTIONS
                   if (row["example_id"], action) in records]
        if not present:
            continue
        direct = records.get((row["example_id"], "direct"))
        if direct is None:
            raise ValueError("Saved branch has no fresh direct record")
        with Image.open(manifest.parent / row["image_path"]) as raw:
            source = ImageOps.exif_transpose(raw).convert("RGB")
        for record in present:
            expected = {key: row[key] for key in ("example_id", "image_id", "source_id", "question")}
            expected.update(target_answer=row["answer"], source_image_sha256=row["image_sha256"])
            expected["observed_direct_response_sha256"] = hashlib.sha256(direct["response"].encode("utf-8")).hexdigest()
            _, images, geometry = build_request(source, row["question"], record["action"], config, direct["response"])
            expected.update(geometry)
            scores = score_response(record["response"], row["answer"])
            expected.update(scores)
            expected.update(score_official(scores["predicted_answer"] or "", row["answer"]))
            expected["image_grid_thw"] = [[1, image.height // 16, image.width // 16] for image in images]
            expected["visual_tokens"] = sum(t * h * w // 4 for t, h, w in expected["image_grid_thw"])
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Saved record content mismatch: {row['example_id']}/{record['action']}")
            if not record["answer_prefix_prefilled"] or record["response"] != "ANSWER:" + record["raw_continuation"]:
                raise ValueError("Saved answer continuation/prefix mismatch")
            if record["generated_tokens"] < 1 or record["generated_tokens"] > config["answer_max_tokens"]:
                raise ValueError("Saved generation length outside contract")
            if record["elapsed_s"] <= 0 or record["input_tokens"] <= record["visual_tokens"]:
                raise ValueError("Invalid saved measurement")
        for region in REGIONS:
            check_pair(records, row["example_id"], region)


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
        if json.loads(meta_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("Run identity changed; use a new output directory")
    else:
        metadata = {**identity, **runtime, "fingerprint": fingerprint, "base_code_sha256": code_digest(),
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "experiment": "history_context_v4", "development_only": True,
                    "cost_definition": "Synchronized full path including PNG decode, view construction, RGB hashes, processor, transfers, generation, decoding and logprob extraction. PDF rendering, model load, warmup and logging excluded; rendering recorded separately. Primary decision-state policy adds fresh direct cost to every regional branch and repeat. Standalone fresh/placeholder cost excludes direct; actual/repeat includes direct. No cross-call KV cache reuse.",
                    "warmup": "Six synthetic calls: direct/actual_native_tl at landscape, square and portrait shapes; two output tokens each. Nonexhaustive shape coverage.",
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
        for action in ("direct", "actual_native_tl"):
            messages, images, _ = build_request(source, "What color is the image?", action, config, "ANSWER: gray")
            backend.generate(messages, images, 2, answer_prefix=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with record_path.open("a", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            alternatives = list(ACTIONS[1:])
            item_seed = int(hashlib.sha256(f'{config["seed"]}:{row["example_id"]}'.encode()).hexdigest()[:16], 16)
            random.Random(item_seed).shuffle(alternatives)
            for action in ["direct", *alternatives]:
                key = row["example_id"], action
                if key in completed:
                    continue
                initial = completed.get((row["example_id"], "direct"), {}).get("response")
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    action_started = time.perf_counter()
                    with Image.open(manifest.parent / row["image_path"]) as raw:
                        source = ImageOps.exif_transpose(raw).convert("RGB")
                    messages, images, geometry = build_request(source, row["question"], action, config, initial)
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
                              "action": action, "status": "ok", **result, **geometry, **scores, "elapsed_s": elapsed,
                              "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
                    record["observed_direct_response_sha256"] = hashlib.sha256(
                        (result["response"] if action == "direct" else initial).encode("utf-8")).hexdigest()
                    if action not in ("direct", "highres"):
                        direct_hash = completed[(row["example_id"], "direct")]["image_rgb_sha256"][0]
                        if record["image_rgb_sha256"][0] != direct_hash:
                            raise ValueError("Follow-up overview differs from direct")
                        if action == "repeat" and record["image_rgb_sha256"][1] != direct_hash:
                            raise ValueError("Repeat is not an exact overview copy")
                    completed[key] = record
                    for region in ("tl", "tr", "bl", "br"):
                        check_pair(completed, row["example_id"], region)
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
