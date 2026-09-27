"""Resumable paired experiments, immutable run identity, synchronized costs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import random
import time

from .actions import build_request, grade


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_pilot(manifest_path: Path, model_dir: Path, config_path: Path, output_dir: Path, limit: int | None = None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from PIL import Image, ImageOps
    from .backend import QwenBackend
    from .data import validate_manifest

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("do_sample") is not False or config.get("dtype") != "bfloat16":
        raise ValueError("this version supports deterministic greedy BF16 experiments only")
    if config.get("short_answer_prefix") != "ANSWER:":
        raise ValueError("short-answer prefix must be the documented ANSWER: contract")
    actions = config["actions"]
    if len(set(actions)) != len(actions) or "direct" not in actions:
        raise ValueError("unique actions including direct are required")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    rows = validate_manifest(manifest_path)
    rows = rows[:limit] if limit else rows
    model_meta = json.loads((model_dir / "lookagain-model.json").read_text(encoding="utf-8"))
    if model_meta["model_id"] != config["model_id"]:
        raise ValueError("model directory and config disagree")
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_dir.mkdir(parents=True, exist_ok=True)
    packages = {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "transformers", "datasets", "Pillow", "numpy", "huggingface-hub"]}
    runtime = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "packages": packages, "python": platform.python_version(), "platform": platform.platform()}
    identity = {"config": config, "manifest_sha256": sha256_file(manifest_path), "code_sha256": code_digest(), "model": model_meta, "runtime": runtime, "example_ids": [row["example_id"] for row in rows]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    metadata_path = output_dir / "run.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("run identity changed; use a new output directory")
    else:
        metadata = {**identity, "fingerprint": fingerprint, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **runtime, "cost_definition": "warm synchronized wall time; action construction, image decode/resize, processor, transfers, generation, decoding and batched logprob extraction; no cross-call KV caching; model load/warmup excluded", "development_only": True}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    records_path = output_dir / "records.jsonl"
    completed = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            key = (record["example_id"], record["action"])
            if key in completed:
                raise ValueError(f"duplicate completed action: {key}")
            completed[key] = record
    expected = {(row["example_id"], action) for row in rows for action in actions}
    if set(completed) - expected:
        raise ValueError("records contain unexpected examples/actions")
    if expected <= completed.keys():
        print("All requested actions already complete.", flush=True)
        return records_path
    backend = QwenBackend(model_dir, config)
    # Warm each relevant image/token shape without using benchmark answers.
    warm_image = Image.new("RGB", (800, 600), (128, 128, 128))
    for warm_action in ("direct", "highres", "crop_tl"):
        msg, imgs, _ = build_request(warm_image, "What color is the image?", warm_action, config, "ANSWER: gray")
        backend.generate(msg, imgs, 2, answer_prefix=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for index, row in enumerate(rows):
        alternatives = [a for a in actions if a != "direct"]
        item_seed = int(hashlib.sha256(f'{seed}:{row["example_id"]}'.encode()).hexdigest()[:16], 16)
        random.Random(item_seed).shuffle(alternatives)
        for action in ["direct", *alternatives]:
            key = (row["example_id"], action)
            if key in completed:
                continue
            initial = completed.get((row["example_id"], "direct"), {}).get("response")
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                action_start = time.perf_counter()
                with Image.open(manifest_path.parent / row["image_path"]) as raw:
                    source = ImageOps.exif_transpose(raw).convert("RGB")
                messages, images, geometry = build_request(source, row["question"], action, config, initial)
                budget = config["think_max_tokens"] if action == "think" else config["answer_max_tokens"]
                result = backend.generate(messages, images, budget, answer_prefix=action != "think")
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - action_start
                record = {
                    "example_id": row["example_id"], "image_id": row["image_id"], "question": row["question"],
                    "target_answer": row["answer"], "action": action, "status": "ok", **result, **geometry,
                    **grade(result["response"], row["answer"]), "elapsed_s": elapsed,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                }
                with records_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                completed[key] = record
                print(f'[{index+1}/{len(rows)}] {action}: {record["predicted_answer"]!r} correct={record["correct"]} {elapsed:.2f}s ({len(completed)}/{len(expected)} actions)', flush=True)
            except Exception as error:
                with (output_dir / "errors.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"example_id": row["example_id"], "action": action, "error": repr(error)}) + "\n")
                raise
    print(f"Completed in {(time.perf_counter()-start)/60:.1f} minutes after warmup.", flush=True)
    return records_path
