"""CPU-only serialized-template and tensor checks; no model answers or weights."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoProcessor

from history_context_core import ACTIONS, HISTORIES, FIDELITIES, REGIONS, PLACEHOLDER, build_request
from native_detail_core import build_request as build_v3_request


def audit(manifest, model_dir, output):
    config = dict(base_visual_tokens=1024, crop_visual_tokens=1024, highres_visual_tokens=4096)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    checks = []
    for row in rows:
        source = Image.open(manifest.parent / row["image_path"]).convert("RGB")
        encoded = {}
        for action in ACTIONS[3:]:
            messages, images, geo = build_request(source, row["question"], action, config, "ANSWER: SENTINEL_NO_TARGET_81739")
            serialized = processor.apply_chat_template([*messages, {"role": "assistant", "content": "ANSWER:"}],
                                                       tokenize=False, continue_final_message=True)
            inputs = processor(text=[serialized], images=images, return_tensors="pt")
            grids = inputs["image_grid_thw"].tolist()
            assert grids == [[1, image.height // 16, image.width // 16] for image in images]
            assert serialized.count("<|vision_start|>") == 2
            assert serialized.endswith("ANSWER:")
            if action.startswith("actual_"):
                old, _, _ = build_v3_request(source, row["question"], action.removeprefix("actual_"), config, "ANSWER: SENTINEL_NO_TARGET_81739")
                old_serialized = processor.apply_chat_template([*old, {"role": "assistant", "content": "ANSWER:"}],
                                                               tokenize=False, continue_final_message=True)
                assert serialized == old_serialized
            else:
                assert "SENTINEL_NO_TARGET_81739" not in serialized
            if action.startswith("placeholder_"):
                assert "<|im_start|>assistant\n" + PLACEHOLDER + "<|im_end|>" in serialized
            if action.startswith("fresh_"):
                assert serialized.count("<|im_start|>assistant") == 1
            encoded[action] = inputs
            checks.append(dict(example_id=row["example_id"], action=action, image_grid_thw=grids,
                               input_tokens=inputs["input_ids"].shape[-1], image_rgb_sha256=geo["image_rgb_sha256"],
                               serialized_sha256=hashlib.sha256(serialized.encode()).hexdigest()))
        for region in REGIONS:
            for history in HISTORIES:
                native = encoded[f"{history}_native_{region}"]
                degraded = encoded[f"{history}_degraded_{region}"]
                assert torch.equal(native["input_ids"], degraded["input_ids"])
                first_count = int(native["image_grid_thw"][0].prod())
                assert torch.equal(native["pixel_values"][:first_count], degraded["pixel_values"][:first_count])
                assert not torch.equal(native["pixel_values"][first_count:].bfloat16(), degraded["pixel_values"][first_count:].bfloat16())
            for fidelity in FIDELITIES:
                reference = encoded[f"actual_{fidelity}_{region}"]
                for history in HISTORIES[1:]:
                    assert torch.equal(reference["pixel_values"], encoded[f"{history}_{fidelity}_{region}"]["pixel_values"])
        print(f"Processor checks passed for {row['example_id']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(status="passed", model_inference=False, reports=len(rows), processor_calls=len(checks),
        scope="Smoke-source CPU tensors and chat serialization only; no production activation capture.",
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), records=checks), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifest.resolve(), args.model_dir.resolve(), args.output.resolve())
