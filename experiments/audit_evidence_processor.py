"""CPU-only V5 processor audit before model inference; no answer generation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor
from evidence_availability import validate_manifest
from evidence_availability_core import ACTIONS, BUDGETS, build_request


def audit(manifest, model_dir, output):
    rows = validate_manifest(manifest)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    config = dict(crop_visual_tokens=1024, highres_visual_tokens=4096)
    checks = []
    for row in rows:
        source = Image.open(manifest.parent / row["image_path"]).convert("RGB")
        encoded = {}
        for action in ACTIONS:
            messages, images, geo = build_request(source, row["question"], action, config, row["roi_pixels"])
            serialized = processor.apply_chat_template([*messages, {"role":"assistant", "content":"ANSWER:"}], tokenize=False, continue_final_message=True)
            inputs = processor(text=[serialized], images=images, return_tensors="pt")
            grids = inputs["image_grid_thw"].tolist()
            assert grids == [[1, image.height//16, image.width//16] for image in images]
            assert serialized.count("<|vision_start|>") == len(images)
            assert serialized.count("<|im_start|>assistant") == 1 and serialized.endswith("ANSWER:")
            encoded[action] = inputs
            checks.append(dict(example_id=row["example_id"], action=action, image_grid_thw=grids,
                input_tokens=inputs["input_ids"].shape[-1], image_rgb_sha256=geo["image_rgb_sha256"],
                serialized_sha256=hashlib.sha256(serialized.encode()).hexdigest()))
        native_tensors = []
        for budget in BUDGETS:
            native, degraded = encoded[f"native_{budget}"], encoded[f"degraded_{budget}"]
            first_count = int(native["image_grid_thw"][0].prod())
            assert torch.equal(native["input_ids"], degraded["input_ids"])
            assert torch.equal(native["pixel_values"][:first_count], degraded["pixel_values"][:first_count])
            assert not torch.equal(native["pixel_values"][first_count:].bfloat16(), degraded["pixel_values"][first_count:].bfloat16())
            assert torch.equal(native["pixel_values"][:first_count], encoded[f"direct_{budget}"]["pixel_values"])
            native_tensors.append(native["pixel_values"][first_count:])
        assert all(torch.equal(native_tensors[0], value) for value in native_tensors[1:])
        print(f"Processor checks passed for {row['example_id']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(status="passed", model_inference=False, reports=len(rows), processor_calls=len(checks),
        scope="CPU tensors and chat serialization only; no production activation capture.",
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), records=checks), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    audit(args.manifest.resolve(),args.model_dir.resolve(),args.output.resolve())
