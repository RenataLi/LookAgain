"""CPU-only four-request preflight; no model weights, generation or scoring.

Accepts pending engineering candidates or final manifests. This checks request
construction, not semantic eligibility or complete corpus source independence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT / "src"))
import evidence_availability_core as core

ACTIONS = ("direct_256", "native_256", "degraded_256", "highres")
FROZEN_FILES = {
    "experiments/evidence_availability_core.py": "b42e805a4bdbe8386aacdf4c4f768cecfb27e97406d43715756b22655fb36df1",
    "experiments/native_detail_core.py": "f3ef26888a2453d08a76c8436f5534ec7670c2f6f7ad9a9245924affdf4bd398",
    "experiments/history_context_core.py": "d5e9775e9454e3b1643612e05f15ce66d14cb9953ed0e74b71802dd536fb1990",
    "src/lookagain/actions.py": "3277f7c95d26d7e45fd2ed889dafc3c9b507e675456644e25a8f19d6d6bbcd41",
    "src/lookagain/backend.py": "07d0225d525a385b7456b7c76b3a58003eff4365e5d76d465b53605b6344aff1",
}
ANSWER = "Answer by copying the relevant single text span from the page, including any units. Do not explain."
FRESH = "The first image is the full page. The second image is an enlarged region of the first image."
SYSTEM = "You answer questions accurately based on the provided images."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chat_hash(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_config(config):
    for key, expected in (("crop_visual_tokens", 1024), ("highres_visual_tokens", 4096)):
        require(type(config.get(key)) is int and config[key] == expected, f"Unexpected {key}")
    if "short_answer_prefix" in config:
        require(config["short_answer_prefix"] == "ANSWER:", "Only frozen ANSWER: prefill is supported")
    # A full frozen v5 config may be supplied, but ONLY the four ACTIONS run.
    # No answer, OCR or extra config field is passed as model language input.


def expected_messages(question, regional):
    text = (FRESH + "\n" if regional else "") + question + "\n" + ANSWER
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
        [{"type": "image"}] * (2 if regional else 1) + [{"type": "text", "text": text}]}]


def image_equal(first, second):
    return first.size == second.size and first.mode == second.mode and first.tobytes() == second.tobytes()


def _view_checks(image, original_size, budget):
    require(image.mode == "RGB", "Non-RGB request view")
    require(all(x % 32 == 0 for x in image.size), "View is not on the 32-pixel grid")
    require(all(x <= original for x, original in zip(image.size, original_size)), "Source view was upsampled")
    tokens = image.width * image.height // (32 * 32)
    require(64 <= tokens <= budget, "View violates budget or processor minimum-pixel guard")
    return tokens


def token_runs(ids, token_id):
    runs, current = [], 0
    for value in ids:
        if value == token_id:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def tensor_hash(value):
    import torch
    data = value.detach().cpu().contiguous()
    return hashlib.sha256(f"{data.dtype}:{list(data.shape)}:".encode() +
                          data.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_difference(first, second):
    import torch
    require(first.shape == second.shape, "Paired tensor shapes differ")
    delta = first.float() - second.float()
    count = int(torch.count_nonzero(delta).item())
    return {"identical": bool(torch.equal(first, second)), "elements": first.numel(),
            "different_elements": count, "different_fraction": count / first.numel(),
            "mean_absolute_difference": float(delta.abs().mean()),
            "max_absolute_difference": float(delta.abs().max())}


def process_request(processor, messages, images):
    import torch
    patch, merge = processor.image_processor.patch_size, processor.image_processor.merge_size
    require(patch == 16 and merge == 2, "Unexpected frozen processor patch/merge factor")
    continued = [*messages, {"role": "assistant", "content": "ANSWER:"}]
    prompt = processor.apply_chat_template(continued, tokenize=False, continue_final_message=True)
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    inputs.pop("token_type_ids", None)  # Same input handling as the frozen backend.
    require(all(x.device.type == "cpu" for x in inputs.values() if isinstance(x, torch.Tensor)), "Non-CPU processor output")
    grids = inputs["image_grid_thw"].tolist()
    expected_grids = [[1, image.height // patch, image.width // patch] for image in images]
    require(grids == expected_grids, "Processor resized, dropped, reordered or mis-gridded an image")
    patch_counts = [math.prod(grid) for grid in grids]
    require(inputs["pixel_values"].shape[0] == sum(patch_counts), "Pixel tensor patch rows do not cover every image")
    token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    visual_runs = token_runs(inputs["input_ids"][0].tolist(), token_id)
    require(visual_runs == [count // (merge * merge) for count in patch_counts], "Image-token blocks do not match ordered image grids")
    summary = {"input_tokens": int(inputs["input_ids"].shape[-1]), "image_grid_thw": grids,
               "visual_tokens": sum(visual_runs), "image_token_runs": visual_runs,
               "pixel_values_shape": list(inputs["pixel_values"].shape),
               "pixel_values_dtype": str(inputs["pixel_values"].dtype),
               "formatted_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
               "tensor_sha256": {key: tensor_hash(value) for key, value in inputs.items() if isinstance(value, torch.Tensor)}}
    return inputs, summary


def check_requests(source, question, roi, config, processor=None):
    """Check one candidate, without accessing its answer/annotation text."""
    validate_config(config)
    require(source.mode == "RGB", "Source must be the verified RGB page render")
    box = core.validate_roi(roi, source.size)
    x0, y0, x1, y1 = box
    require((x1-x0)*(y1-y0) <= 0.25*source.width*source.height, "ROI exceeds quarter-page area")
    require(isinstance(question, str) and bool(question.strip()), "Missing question")
    overview = source.resize(core.bounded_size(source.size, 256), Image.Resampling.BICUBIC)
    highres = source.resize(core.bounded_size(source.size, 4096), Image.Resampling.BICUBIC)
    crop_size = core.bounded_size((x1-x0, y1-y0), 1024)
    projected = (x0*overview.width/source.width, y0*overview.height/source.height,
                 x1*overview.width/source.width, y1*overview.height/source.height)
    expected_native = source.crop(box).resize(crop_size, Image.Resampling.BICUBIC)
    expected_degraded = overview.resize(crop_size, Image.Resampling.BICUBIC, box=projected)
    expected = {"direct_256": [overview], "native_256": [overview, expected_native],
                "degraded_256": [overview, expected_degraded], "highres": [highres]}
    inputs, rows = {}, {}
    for action in ACTIONS:
        messages, images, geometry = core.build_request(source, question, action, config, list(box))
        regional = action in ("native_256", "degraded_256")
        # Exact template comparison permits incidental gold words in the original question.
        require(messages == expected_messages(question, regional), f"{action}: unexpected chat content/history")
        require(len(images) == len(expected[action]), f"{action}: wrong image count")
        require(all(image_equal(x, y) for x, y in zip(images, expected[action])), f"{action}: incorrect crop, projection or overview pixels")
        require(geometry["messages_sha256"] == chat_hash(messages), f"{action}: incorrect chat digest")
        require(geometry["image_rgb_sha256"] == [core.image_digest(x) for x in images], f"{action}: incorrect image digest")
        require(geometry["source_size"] == list(source.size) and geometry["overview_size"] == list(images[0].size), f"{action}: wrong source/overview size metadata")
        require(geometry["history_mode"] == "fresh" and geometry["previous_answer_in_prompt"] is False, f"{action}: hidden answer history")
        require(geometry["condition"] == action.split("_")[0] and geometry["overview_budget"] == (4096 if action == "highres" else 256), f"{action}: wrong condition/budget metadata")
        require(geometry["roi_used"] is regional and geometry["roi_localizer"] == "reference_annotation_privileged", f"{action}: incorrect ROI privilege metadata")
        require(geometry["resampling"] == "BICUBIC" and geometry["crop_visual_token_budget"] == 1024, f"{action}: incorrect resampling/budget metadata")
        visual = [_view_checks(images[0], source.size, 4096 if action == "highres" else 256)]
        if regional:
            require(geometry["source_roi_pixels"] == list(box), f"{action}: incorrect source ROI metadata")
            require(geometry["projected_overview_roi_pixels"] == list(projected), f"{action}: incorrect projection metadata")
            require(geometry["additional_size"] == list(crop_size), f"{action}: incorrect crop size metadata")
            require(geometry["crop_box_normalized"] == [x0/source.width, y0/source.height, x1/source.width, y1/source.height], f"{action}: incorrect normalized ROI")
            require(geometry["additional_view_kind"] == ("source_render_roi" if action == "native_256" else "overview_derived_roi"), f"{action}: incorrect fidelity metadata")
            visual.append(_view_checks(images[1], (x1-x0, y1-y0), 1024))
        else:
            require(geometry["roi_used"] is False and geometry["additional_size"] is None, f"{action}: unexpected ROI use")
        rows[action] = {"geometry": geometry, "image_sizes": [list(x.size) for x in images], "visual_tokens_from_dimensions": visual}
        if processor is not None:
            inputs[action], rows[action]["processor"] = process_request(processor, messages, images)
    warnings = []
    pixel_same = image_equal(expected_native, expected_degraded)
    if pixel_same:
        warnings.append("native_degraded_RGB_identical_keep_example_no_automatic_exclusion")
    pair = {"messages_identical": rows["native_256"]["geometry"]["messages_sha256"] == rows["degraded_256"]["geometry"]["messages_sha256"],
            "native_degraded_rgb_identical": pixel_same}
    require(pair["messages_identical"], "Paired prompts differ")
    if processor is not None:
        import torch
        native, degraded, direct = (inputs[x] for x in ("native_256", "degraded_256", "direct_256"))
        for key in ("input_ids", "attention_mask", "image_grid_thw"):
            require(torch.equal(native[key], degraded[key]), f"Paired processor {key} differs")
        lengths = [math.prod(grid) for grid in native["image_grid_thw"].tolist()]
        nv, dv = (x["pixel_values"].split(lengths, dim=0) for x in (native, degraded))
        require(torch.equal(nv[0], dv[0]) and torch.equal(nv[0], direct["pixel_values"]), "Overview tensor changed between direct/native/degraded")
        second = tensor_difference(nv[1], dv[1])
        bf16 = tensor_difference(nv[1].to(torch.bfloat16), dv[1].to(torch.bfloat16))
        pair.update(input_ids_attention_grids_identical=True, overview_tensor_identical=True,
                    second_view_difference=second, second_view_bf16_cast_difference=bf16)
        if second["identical"]:
            warnings.append("native_degraded_processor_tensors_identical_keep_example_no_automatic_exclusion")
        elif bf16["identical"]:
            warnings.append("native_degraded_differences_vanish_after_bf16_cast_keep_example")
    return {"actions": rows, "pair": pair, "warnings": warnings}


def check_manifest(manifest, config, processor=None, limit=None):
    """Bind a source manifest; subset coverage and pending semantic state remain explicit."""
    manifest = Path(manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(bool(rows), "Manifest is empty")
    require(len({row["example_id"] for row in rows}) == len(rows), "Duplicate manifest example ID")
    if limit is not None:
        require(type(limit) is int and limit > 0, "limit must be a positive integer")
    selected = rows if limit is None else rows[:limit]
    results = []
    for row in selected:
        result = {"example_id": row["example_id"], "source_id": row.get("source_cluster_id", row.get("source_id")),
                  "source_semantic_audit_status": row.get("source_semantic_audit_status", "not_supplied")}
        try:
            relative = Path(row["image_path"])
            require(not relative.is_absolute(), "Expected manifest-relative image_path")
            image_path = (manifest.parent / relative).resolve()
            require(image_path.is_relative_to(manifest.parent), "Image path escapes manifest directory")
            image_hash = sha_file(image_path)
            require(image_hash == row["image_sha256"], "Source image file hash mismatch")
            if "rendered_png_sha256" in row:
                require(image_hash == row["rendered_png_sha256"], "Rendered PNG hash mismatch")
            with Image.open(image_path) as raw:
                source = ImageOps.exif_transpose(raw).convert("RGB")
            if "pixel_dimensions" in row:
                require(list(source.size) == row["pixel_dimensions"], "Manifest raster dimensions mismatch")
            if "width" in row or "height" in row:
                require(list(source.size) == [row.get("width"), row.get("height")], "Manifest width/height mismatch")
            result.update(check_requests(source, row["question"], row["roi_pixels"], config, processor))
            result.update(status="PASS", source_image_sha256=image_hash, source_rgb_sha256=core.image_digest(source))
        except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
            result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        results.append(result)
    return {"manifest_rows": len(rows), "checked_rows": len(selected), "complete_manifest_coverage": len(selected) == len(rows),
            "distinct_source_ids_checked": len({x["source_id"] for x in results if x["source_id"] is not None}),
            "selection": "All manifest rows in original order" if limit is None else f"First {limit} manifest rows, independent of labels/model outcomes",
            "source_semantic_status_counts": dict(Counter(x["source_semantic_audit_status"] for x in results)),
            "rows": results, "structural_failures": sum(x["status"] == "FAIL" for x in results),
            "warning_counts": dict(Counter(w for x in results for w in x.get("warnings", [])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--model-dir", type=Path, help="Optional LOCAL processor directory; no model weights loaded")
    parser.add_argument("--limit", type=int, help="Optional deterministic first-N subset; default checks all rows")
    args = parser.parse_args()
    actual_hashes = {name: sha_file(PROJECT / name) for name in FROZEN_FILES}
    require(actual_hashes == FROZEN_FILES, "Frozen v5 request/backend source hashes changed")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    processor, processor_files, runtime = None, {}, {"Pillow": importlib.metadata.version("Pillow")}
    if args.model_dir is not None:
        require(args.model_dir.is_dir(), "Processor must be an existing local directory")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoProcessor
        torch.set_num_threads(4)
        processor = AutoProcessor.from_pretrained(str(args.model_dir.resolve()), local_files_only=True, trust_remote_code=False)
        for name in ("preprocessor_config.json", "processor_config.json", "tokenizer_config.json", "chat_template.json", "chat_template.jinja", "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json", "config.json"):
            if (args.model_dir / name).is_file():
                processor_files[name] = sha_file(args.model_dir / name)
        runtime.update({name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")})
    report = check_manifest(args.manifest, config, processor, args.limit)
    report.update(
        status="FAIL" if report["structural_failures"] else ("PASS_WITH_WARNINGS" if report["warning_counts"] else "PASS"),
        created_utc=datetime.now(timezone.utc).isoformat(), actions=list(ACTIONS),
        processor_checked=processor is not None,
        execution={"device": "cpu", "model_weights_loaded": False, "generation_calls": 0, "model_outputs_read": False,
                   "processor_local_files_only": True if processor else None, "trust_remote_code": False},
        bindings={"manifest_sha256": sha_file(args.manifest), "config_sha256": sha_file(args.config),
                  "config": config, "preflight_sha256": sha_file(__file__), "frozen_sources": actual_hashes,
                  "processor_files": processor_files}, runtime=runtime,
        limitations=["Technical preflight does not approve semantic quality, final cohort eligibility or source independence.",
                     "Inputs are reconstructed on CPU; no model execution activations or answers are inspected.",
                     "Identical native/degraded pixels or tensors remain explicit warnings; no example is removed and no ROI is changed.",
                     "BF16 comparison is a tensor cast, not an internal vision-embedding measurement."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checked_rows": report["checked_rows"],
                      "processor_checked": report["processor_checked"], "structural_failures": report["structural_failures"]}))
    if report["structural_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
