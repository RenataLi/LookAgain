"""Request contract checks with synthetic sources; no model weights/generation."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_dude_requests", Path(__file__).resolve().parents[1] / "experiments/check_dude_requests.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)
CONFIG = {"crop_visual_tokens": 1024, "highres_visual_tokens": 4096, "short_answer_prefix": "ANSWER:"}
ROI = [57, 63, 401, 343]


@pytest.fixture(scope="module")
def source():
    image = Image.new("RGB", (1025, 1537), "white")
    draw = ImageDraw.Draw(image)
    for x in range(0, image.width, 7):
        draw.line((x, 0, x, image.height), fill=(x % 256, 10, 90), width=2)
    draw.text((82, 98), "A source detail 123", fill="black")
    return image


class FakeProcessor:
    """Known per-patch CPU encoding to exercise image-order/shape checks."""
    image_processor = SimpleNamespace(patch_size=16, merge_size=2)
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 777)
    image_token = "image"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "continue_final_message": True}
        assert messages[-1] == {"role": "assistant", "content": "ANSWER:"}
        return json.dumps(messages, sort_keys=True)

    def __call__(self, *, text, images, return_tensors):
        import torch
        assert len(text) == 1 and return_tensors == "pt"
        grids, blocks, ids = [], [], [3]
        for image in images:
            w, h = image.width // 16, image.height // 16
            grids.append([1, h, w])
            reduced = image.resize((w, h), Image.Resampling.BICUBIC)
            blocks.append(torch.frombuffer(bytearray(reduced.tobytes()), dtype=torch.uint8).reshape(-1, 3).float())
            ids.extend([777] * (w*h//4) + [9])
        return {"image_grid_thw": torch.tensor(grids), "pixel_values": torch.cat(blocks),
                "input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.int64)}


def test_odd_coordinates_projection_and_no_upsample_all_four_requests(source):
    result = check.check_requests(source, "Which number 123?", ROI, {**CONFIG, "answer": "123", "ocr": "unused secret"})
    assert list(result["actions"]) == list(check.ACTIONS)
    assert result["pair"]["messages_identical"]
    assert not result["pair"]["native_degraded_rgb_identical"]
    native = result["actions"]["native_256"]
    degraded = result["actions"]["degraded_256"]
    assert native["geometry"]["projected_overview_roi_pixels"] == degraded["geometry"]["projected_overview_roi_pixels"]
    assert native["image_sizes"] == degraded["image_sizes"]
    assert result["actions"]["highres"]["visual_tokens_from_dimensions"][0] < 4096


def test_constant_source_zero_effective_pixel_contrast_is_only_a_warning():
    result = check.check_requests(Image.new("RGB", (1025, 1537), "white"), "Question?", ROI, CONFIG)
    assert result["pair"]["native_degraded_rgb_identical"]
    assert result["warnings"] == ["native_degraded_RGB_identical_keep_example_no_automatic_exclusion"]


def test_injected_label_or_history_is_caught_by_exact_chat_not_substring(source, monkeypatch):
    original = check.core.build_request
    def corrupt(*args):
        messages, images, geometry = original(*args)
        messages[-1]["content"][-1]["text"] += "\nReference: 123"
        geometry["messages_sha256"] = check.chat_hash(messages)
        return messages, images, geometry
    monkeypatch.setattr(check.core, "build_request", corrupt)
    with pytest.raises(ValueError, match="unexpected chat"):
        check.check_requests(source, "Which number 123?", ROI, CONFIG)


def test_wrong_crop_pixels_fail_even_if_hash_is_updated(source, monkeypatch):
    original = check.core.build_request
    def corrupt(*args):
        messages, images, geometry = original(*args)
        if args[2] == "native_256":
            images[1] = Image.new("RGB", images[1].size, "red")
            geometry["image_rgb_sha256"] = [check.core.image_digest(x) for x in images]
        return messages, images, geometry
    monkeypatch.setattr(check.core, "build_request", corrupt)
    with pytest.raises(ValueError, match="incorrect crop"):
        check.check_requests(source, "Q?", ROI, CONFIG)


def test_cpu_processor_pair_checks_and_real_token_blocks(source):
    result = check.check_requests(source, "Q?", ROI, CONFIG, FakeProcessor())
    pair = result["pair"]
    assert pair["input_ids_attention_grids_identical"] and pair["overview_tensor_identical"]
    assert pair["second_view_difference"]["different_elements"] > 0
    for action in check.ACTIONS:
        row = result["actions"][action]
        assert row["processor"]["image_token_runs"] == row["visual_tokens_from_dimensions"]


def test_cpu_identical_second_view_tensor_is_not_an_exclusion():
    result = check.check_requests(Image.new("RGB", (1025, 1537), "white"), "Q?", ROI, CONFIG, FakeProcessor())
    assert result["pair"]["second_view_difference"]["identical"]
    assert "native_degraded_processor_tensors_identical_keep_example_no_automatic_exclusion" in result["warnings"]


def test_processor_dropped_second_image_is_structural_failure(source):
    class DroppingProcessor(FakeProcessor):
        def __call__(self, **kwargs):
            kwargs["images"] = kwargs["images"][:1]
            return super().__call__(**kwargs)
    with pytest.raises(ValueError, match="dropped"):
        check.check_requests(source, "Q?", ROI, CONFIG, DroppingProcessor())


def test_budget_changes_are_rejected(source):
    with pytest.raises(ValueError, match="highres_visual_tokens"):
        check.check_requests(source, "Q?", ROI, {**CONFIG, "highres_visual_tokens": 2048})
    with pytest.raises(ValueError, match="quarter-page"):
        check.check_requests(source, "Q?", [0, 0, 1024, 1024], CONFIG)


def test_manifest_binds_source_keeps_pending_status_and_subset_denominator(tmp_path, source):
    path = tmp_path / "page.png"
    source.save(path)
    row = {"example_id": "a", "source_cluster_id": "source_a", "image_path": "page.png",
           "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "pixel_dimensions": list(source.size), "roi_pixels": ROI, "question": "Q?",
           "source_semantic_audit_status": "pending", "validated_primary_answers": []}
    manifest = tmp_path / "manifest.jsonl"
    rows = [row, {**row, "example_id": "b", "source_cluster_id": "source_b"}]
    manifest.write_text("\n".join(json.dumps(x) for x in rows))
    report = check.check_manifest(manifest, CONFIG, limit=1)
    assert report["manifest_rows"] == 2 and report["checked_rows"] == 1
    assert not report["complete_manifest_coverage"]
    assert report["source_semantic_status_counts"] == {"pending": 1}
    assert report["structural_failures"] == 0
    rows[0]["image_sha256"] = "not_the_source_hash"
    manifest.write_text("\n".join(json.dumps(x) for x in rows))
    failed = check.check_manifest(manifest, CONFIG)
    assert failed["structural_failures"] == 1
    assert "hash mismatch" in failed["rows"][0]["error"]


def test_image_token_run_order_is_preserved():
    assert check.token_runs([1, 7, 7, 2, 7, 7, 7, 2], 7) == [2, 3]
