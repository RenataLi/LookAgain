"""Post-run CPU-only processor reconstruction, without model weights or generation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT / "src"))

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor
from native_detail_core import build_request
from native_detail import experiment_digest
from lookagain.runner import sha256_file

torch.set_num_threads(4)
parser = argparse.ArgumentParser(description=__doc__)
for name in ("run", "manifest", "model-dir", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
run, manifest, model_dir, output = args.run.resolve(), args.manifest.resolve(), args.model_dir.resolve(), args.output.resolve()
metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
completion = json.loads((run / "completed.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
sources = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(records) == 1100 and completion["records"] == 1100
assert completion["records_sha256"] == sha256_file(run / "records.jsonl")
assert metadata["manifest_sha256"] == sha256_file(manifest)
assert experiment_digest() == metadata["code_sha256"]
by_key = {(r["example_id"], r["action"]): r for r in records}
processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
image_token = getattr(processor, "image_token", None)
image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token) if image_token else None


def tensor_hash(tensor):
    data = tensor.detach().cpu().contiguous()
    prefix = f"{data.dtype}:{list(data.shape)}:".encode()
    return hashlib.sha256(prefix + data.view(torch.uint8).numpy().tobytes()).hexdigest()


def difference(first, second):
    delta = first.float() - second.float()
    return {"identical": bool(torch.equal(first, second)), "shape": list(first.shape),
            "elements": first.numel(), "different_elements": int(torch.count_nonzero(delta).item()),
            "different_fraction": float(torch.count_nonzero(delta).item() / delta.numel()),
            "mean_absolute_difference": float(delta.abs().mean().item()),
            "max_absolute_difference": float(delta.abs().max().item()),
            "root_mean_square_difference": float(delta.square().mean().sqrt().item())}


def runs_of_image_tokens(ids):
    if image_token_id is None:
        return None
    runs, current = [], 0
    for token in ids[0].tolist():
        if token == image_token_id:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def prepare(source, question, action, initial):
    messages, images, geometry = build_request(source, question, action, metadata["config"], initial)
    continued = [*messages, {"role": "assistant", "content": "ANSWER:"}]
    prompt = processor.apply_chat_template(continued, tokenize=False, continue_final_message=True)
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    assert all(value.device.type == "cpu" for value in inputs.values() if isinstance(value, torch.Tensor))
    return inputs, geometry


pairs = []
for rank, source_row in enumerate(sources[:3], 1):
    assert source_row["example_id"] == metadata["example_ids"][rank - 1]
    image_path = manifest.parent / source_row["image_path"]
    assert sha256_file(image_path) == source_row["image_sha256"]
    with Image.open(image_path) as raw:
        source = ImageOps.exif_transpose(raw).convert("RGB")
    initial = by_key[(source_row["example_id"], "direct")]["response"]
    for region in ("tl", "tr", "bl", "br"):
        native, native_geometry = prepare(source, source_row["question"], "native_" + region, initial)
        degraded, degraded_geometry = prepare(source, source_row["question"], "degraded_" + region, initial)
        grids = native["image_grid_thw"].tolist()
        patch_counts = [t * h * w for t, h, w in grids]
        assert len(grids) == 2 and native["pixel_values"].shape[0] == sum(patch_counts)
        assert degraded["pixel_values"].shape == native["pixel_values"].shape
        native_views = native["pixel_values"].split(patch_counts, dim=0)
        degraded_views = degraded["pixel_values"].split(patch_counts, dim=0)
        first_diff = difference(native_views[0], degraded_views[0])
        second_diff = difference(native_views[1], degraded_views[1])
        second_bf16 = difference(native_views[1].to(torch.bfloat16), degraded_views[1].to(torch.bfloat16))
        record_native = by_key[(source_row["example_id"], "native_" + region)]
        record_degraded = by_key[(source_row["example_id"], "degraded_" + region)]
        checks = {
            "input_ids_identical": bool(torch.equal(native["input_ids"], degraded["input_ids"])),
            "attention_masks_identical": bool(torch.equal(native["attention_mask"], degraded["attention_mask"])),
            "image_grids_identical": bool(torch.equal(native["image_grid_thw"], degraded["image_grid_thw"])),
            "first_view_tensor_identical": first_diff["identical"],
            "second_view_tensor_different": not second_diff["identical"],
            "second_view_tensor_different_after_bf16_cast": not second_bf16["identical"],
            "actual_grid_matches_logged_native": grids == record_native["image_grid_thw"],
            "actual_grid_matches_logged_degraded": degraded["image_grid_thw"].tolist() == record_degraded["image_grid_thw"],
            "actual_input_count_matches_logged_native": native["input_ids"].shape[-1] == record_native["input_tokens"],
            "actual_input_count_matches_logged_degraded": degraded["input_ids"].shape[-1] == record_degraded["input_tokens"],
            "rebuilt_native_chat_hash_matches_logged": native_geometry["messages_sha256"] == record_native["messages_sha256"],
            "rebuilt_degraded_chat_hash_matches_logged": degraded_geometry["messages_sha256"] == record_degraded["messages_sha256"],
            "rebuilt_native_pixel_hashes_match_logged": native_geometry["image_rgb_sha256"] == record_native["image_rgb_sha256"],
            "rebuilt_degraded_pixel_hashes_match_logged": degraded_geometry["image_rgb_sha256"] == record_degraded["image_rgb_sha256"],
            "two_image_token_runs_match_grids": runs_of_image_tokens(native["input_ids"]) == [count // 4 for count in patch_counts],
        }
        pairs.append({
            "selection_rank": rank, "example_id": source_row["example_id"], "source_id": source_row["source_id"],
            "region": region, "source_image_sha256": source_row["image_sha256"],
            "input_keys": list(native.keys()), "input_tokens": int(native["input_ids"].shape[-1]),
            "image_grid_thw": grids, "pixel_values_dtype": str(native["pixel_values"].dtype),
            "pixel_values_shape": list(native["pixel_values"].shape), "unmerged_patch_rows_per_image": patch_counts,
            "image_token_run_lengths": runs_of_image_tokens(native["input_ids"]),
            "checks": checks, "first_view_difference_float32": first_diff,
            "second_view_difference_float32": second_diff, "second_view_difference_after_bf16_cast": second_bf16,
            "tensor_sha256": {
                "first_native": tensor_hash(native_views[0]), "first_degraded": tensor_hash(degraded_views[0]),
                "second_native": tensor_hash(native_views[1]), "second_degraded": tensor_hash(degraded_views[1])},
        })
        print(f"rank={rank} region={region} passed={all(checks.values())} second_MAE={second_diff['mean_absolute_difference']:.6f}", flush=True)

result = {
    "status": "passed" if all(all(pair["checks"].values()) for pair in pairs) else "failed",
    "audit_role": "Post-run bounded engineering check; not prespecified inferential evidence or a model-behavior intervention.",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "selection": "First three rows of the frozen main manifest, all four regions; deterministic rank selection after run completion, without selecting on predictions.",
    "n_sources": 3, "n_native_degraded_pairs": len(pairs), "n_processor_calls": 2 * len(pairs),
    "execution": {"device": "cpu", "torch_threads": 4, "model_weights_loaded": False, "generation_calls": 0,
                  "processor_local_files_only": True, "trust_remote_code": False},
    "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "Pillow", "numpy")},
    "processor_class": type(processor).__name__, "image_token": image_token, "image_token_id": image_token_id,
    "bindings": {"run_fingerprint": metadata["fingerprint"], "records_sha256": sha256_file(run / "records.jsonl"),
                 "manifest_sha256": sha256_file(manifest), "inference_code_sha256": experiment_digest(),
                 "backend_sha256": sha256_file(PROJECT / "src" / "lookagain" / "backend.py"),
                 "audit_script_sha256": sha256_file(__file__),
                 "processor_files": {name: sha256_file(model_dir / name) for name in
                    ("preprocessor_config.json", "processor_config.json", "tokenizer_config.json", "chat_template.json")
                    if (model_dir / name).exists()}},
    "backend_read_only_review": {
        "ordered_images_forwarded": "QwenBackend.generate passes the complete ordered images list to processor(text=[prompt], images=images, return_tensors='pt').",
        "processor_values_forwarded": "The resulting BatchFeature is moved to CUDA; only token_type_ids is removed. All remaining fields, including pixel_values and image_grid_thw, are passed through **inputs to model.generate.",
        "finding": "No application-level slicing, substitution or omission of the second image was found in the frozen backend.",
    },
    "pairs": pairs,
    "limitations": [
        "Inputs are reconstructed after execution with the same local processor and bound source images; tensors or model activations were not captured during the production calls.",
        "The check covers three deterministic source reports and four regions per source, not every input.",
        "Different second-view tensors rule out an identical-input preprocessing failure in these reconstructed cases; they do not establish how strongly the model used either image.",
        "BF16 comparison is a post hoc cast of processor tensors, not a measurement of internal vision embeddings or attention.",
        "Near-identical generated answers and a null quality contrast do not prove that the model ignores images, copies its previous answer, or uses a particular anchoring mechanism.",
    ],
}
output.mkdir(parents=True, exist_ok=True)
(output / "processor_input_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
lines = ["# Post-run processor input audit", "", f"Status: **{result['status']}**. CPU-only reconstruction on the first three frozen-manifest source reports and all four regions: 12 paired comparisons, 24 processor calls, no model weights or generation.",
         "", "The audit repeats the backend's chat template and AutoProcessor calls, including the ANSWER: prefill. Each pixel_values tensor is split into its two image blocks using the unmerged image_grid_thw patch counts.",
         "", "| Rank | Example ID | Region | Same first-view tensor | Different second view | Second-view mean / max absolute difference | Different elements | Same text/grids |",
         "| --- | --- | --- | --- | --- | ---: | ---: | --- |"]
for pair in pairs:
    d, c = pair["second_view_difference_float32"], pair["checks"]
    lines.append(f"| {pair['selection_rank']} | {pair['example_id']} | {pair['region']} | {c['first_view_tensor_identical']} | {c['second_view_tensor_different']} | "
                 f"{d['mean_absolute_difference']:.6f} / {d['max_absolute_difference']:.6f} | {100*d['different_fraction']:.2f}% | "
                 f"{c['input_ids_identical'] and c['image_grids_identical']} |")
lines += ["", "Every reconstructed pair also checks logged chat/pixel hashes, logged token counts, both image-token blocks and differences surviving a BF16 cast. Differences are in normalized processor pixel values, not image-quality or perceptual units.",
          "", "## Frozen backend review", "", result["backend_read_only_review"]["ordered_images_forwarded"],
          result["backend_read_only_review"]["processor_values_forwarded"], result["backend_read_only_review"]["finding"],
          "", "## Scope of the conclusion", ""]
lines += ["- " + limitation for limitation in result["limitations"]]
lines += ["", f"Run fingerprint: `{metadata['fingerprint']}`.", f"Inference code SHA256: `{metadata['code_sha256']}`.",
          "The JSON includes source IDs, tensor shapes/hashes, every check and float32/BF16 difference statistics.", ""]
(output / "processor_input_audit.md").write_text("\n".join(lines), encoding="utf-8")
print("FINAL_STATUS=" + result["status"], flush=True)

