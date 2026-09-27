"""Frozen four-condition follow-up experiment; the original v1 package is unchanged."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lookagain.actions import CROP_BOXES, build_request, grade
from lookagain.runner import code_digest, sha256_file

CONDITIONS = ("named", "neutral", "sham", "frame")
REGIONS = ("tl", "tr", "bl", "br")
LOCATIONS = {"tl": "upper-left", "tr": "upper-right", "bl": "lower-left", "br": "lower-right"}
ACTIONS = ["direct", *(f"{condition}_{region}" for condition in CONDITIONS for region in REGIONS)]
BASE_SOURCE_SHA256 = "b897b6bf0e13c1045b1f1c77a732978ce91007ea87c513f2828547b3424003be"


def experiment_digest():
    """Bind the byte-preserved v1 package and this complete new runner."""
    digest = hashlib.sha256()
    digest.update(code_digest().encode())
    digest.update(Path(__file__).name.encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def image_digest(image):
    return hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()).hexdigest()


def build_condition(source, question, action, config, initial_answer=None):
    """Construct a request from pre-action evidence only, with auditable pixels."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown experiment action: {action}")
    if action == "direct":
        messages, images, geometry = build_request(source, question, "direct", config)
        condition, region = "direct", None
        followup = None
        geometry.update(claimed_crop_box=None, actual_second_view_box=None, additional_view_kind=None)
    else:
        condition, region = action.split("_")
        messages, images, geometry = build_request(source, question, "crop_" + region, config, initial_answer)
        claimed = list(CROP_BOXES["crop_" + region])
        if condition == "sham":
            # Exact copy of the previously shown overview: no new pixels/region.
            images[1] = images[0].copy()
            geometry.update(crop_box_normalized=None, crop_box_pixels=None, crop_size=None)
            actual_box, kind = [0.0, 0.0, 1.0, 1.0], "exact_overview_repeat"
        else:
            actual_box, kind = claimed, "true_crop"
        followup = config["condition_prompts"][condition].format(location=LOCATIONS[region])
        messages[-1]["content"][-1]["text"] = followup
        geometry.update(claimed_crop_box=None if condition == "neutral" else claimed,
                        actual_second_view_box=actual_box, additional_view_kind=kind,
                        assigned_region_box=claimed, additional_size=list(images[1].size))
    geometry.update(condition=condition, region=region, followup_prompt=followup,
                    image_rgb_sha256=[image_digest(image) for image in images])
    return messages, images, geometry


def validate_config(config):
    if config["actions"] != ACTIONS or config["conditions"] != list(CONDITIONS) or config["regions"] != list(REGIONS):
        raise ValueError("The complete fixed 17-action design is required")
    if config["dtype"] != "bfloat16" or config["do_sample"] is not False or config["short_answer_prefix"] != "ANSWER:":
        raise ValueError("Only the frozen BF16 greedy ANSWER: response contract is supported")
    if config["condition_prompts"]["named"] != config["condition_prompts"]["sham"]:
        raise ValueError("Named/sham controls require identical text")
    if set(config["condition_prompts"]) != set(CONDITIONS):
        raise ValueError("All four prompt templates are required")
    if "{location}" in config["condition_prompts"]["neutral"]:
        raise ValueError("Neutral prompt must not disclose crop direction")


def read_records(path):
    completed = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["example_id"], row["action"])
            if key in completed or row["status"] != "ok":
                raise ValueError(f"Duplicate or unsuccessful record: {key}")
            completed[key] = row
    return completed


def run(manifest, model_dir, config_path, output, limit=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from PIL import Image, ImageOps
    from lookagain.backend import QwenBackend
    from lookagain.data import validate_manifest

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if code_digest() != BASE_SOURCE_SHA256:
        raise ValueError("The archived v1 inference package has changed")
    rows = validate_manifest(manifest)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    rows = rows[:limit] if limit else rows
    if not rows:
        raise ValueError("An empty run is invalid")
    model_meta = json.loads((model_dir / "lookagain-model.json").read_text(encoding="utf-8"))
    if model_meta["model_id"] != config["model_id"]:
        raise ValueError("Model and configuration disagree")
    pins = config["protocol"]
    if (sha256_file(manifest) != pins["manifest_sha256"]
            or model_meta["revision"] != pins["model_revision"]):
        raise ValueError("Model/manifest do not match the frozen protocol")
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    packages = {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "transformers", "datasets", "Pillow", "numpy", "huggingface-hub"]}
    runtime = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "packages": packages,
               "python": platform.python_version(), "platform": platform.platform()}
    identity = {"config": config, "manifest_sha256": sha256_file(manifest), "code_sha256": experiment_digest(),
                "model": model_meta, "runtime": runtime, "example_ids": [row["example_id"] for row in rows]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "run.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("Run identity changed; use a new output directory")
    else:
        metadata = {**identity, **runtime, "fingerprint": fingerprint,
                    "base_code_sha256": code_digest(), "experiment_script_sha256": sha256_file(__file__),
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "development_only": True, "experiment": "direction_controls_v2",
                    "warmup": "Synthetic 4:3, 1:1 and 3:4 images; direct and named crop, two output tokens each. Not exhaustive shape coverage.",
                    "cost_definition": "Synchronized full-path action latency including image decode, resize, pixel hashes, processor, transfers, generation, decoding and logprob extraction; excludes loading, warmup and JSONL logging. Follow-up policy cost adds the fresh direct call; no cross-call KV reuse.",
                    "durability": "JSONL flush per action; fsync after each completed image. Strict resume: malformed/duplicate records fail loudly."}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    path = output / "records.jsonl"
    completed = read_records(path)
    expected = {(row["example_id"], action) for row in rows for action in ACTIONS}
    if set(completed) - expected:
        raise ValueError("Unexpected example/action in records")
    if expected <= completed.keys():
        print("All declared actions already complete.", flush=True)
        return
    backend = QwenBackend(model_dir, config)
    for size in ((800, 600), (640, 640), (600, 800)):
        source = Image.new("RGB", size, (128, 128, 128))
        for action in ("direct", "named_tl"):
            messages, images, _ = build_condition(source, "What color is the image?", action, config, "ANSWER: gray")
            backend.generate(messages, images, 2, answer_prefix=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            alternatives = ACTIONS[1:].copy()
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
                    action_started = time.perf_counter()
                    with Image.open(manifest.parent / row["image_path"]) as raw:
                        source = ImageOps.exif_transpose(raw).convert("RGB")
                    messages, images, geometry = build_condition(source, row["question"], action, config, initial)
                    result = backend.generate(messages, images, config["answer_max_tokens"], answer_prefix=True)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - action_started
                    record = {"example_id": row["example_id"], "image_id": row["image_id"], "question": row["question"],
                              "target_answer": row["answer"], "action": action, "status": "ok", **result, **geometry,
                              **grade(result["response"], row["answer"]), "elapsed_s": elapsed,
                              "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed[key] = record
                except Exception as error:
                    with (output / "errors.jsonl").open("a", encoding="utf-8") as errors:
                        errors.write(json.dumps({"example_id": row["example_id"], "action": action, "error": repr(error)}) + "\n")
                    raise
            os.fsync(stream.fileno())
            print(f"[{index+1}/{len(rows)}] {len(completed)}/{len(expected)} actions complete; {(time.perf_counter()-started)/60:.1f} min after warmup", flush=True)
    print(f"Completed {len(completed)} actions in {(time.perf_counter()-started)/60:.1f} minutes after warmup.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.manifest.resolve(), args.model_dir.resolve(), args.config.resolve(), args.output.resolve(), args.limit)


if __name__ == "__main__":
    main()
