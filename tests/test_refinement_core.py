"""CPU-only refinement request contracts; no model, responses, or real labels."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "experiments"), str(PROJECT / "src")]
import refinement_core as core
import region_selection_core as legacy
from native_detail_core import ANSWER_INSTRUCTION

CONFIG = {"base_visual_tokens": 256, "crop_visual_tokens": 1024,
          "highres_visual_tokens": 4096, "seed": 20260927}


@pytest.fixture(scope="module")
def source():
    # Unequal, odd dimensions make transposition, quarter offsets and rounding visible.
    yy, xx = np.indices((1027, 1025))
    pixels = np.stack(((3 * xx + yy) % 256, (xx + 7 * yy) % 256,
                       ((xx // 9) ^ (yy // 13)) % 256), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def equal_images(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.mode == b.mode == "RGB"
        assert a.size == b.size
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("region", range(1, 10))
def test_new_native_changes_only_terminal_instruction_and_its_provenance(source, region):
    question = "Which amount is printed? The question quotes: " + ANSWER_INSTRUCTION
    old_m, old_i, old_g = legacy.build_answer_request(source, question, "native", CONFIG, region)
    new_m, new_i, new_g = core.make_request(source, question, f"new_native_{region}", CONFIG)
    equal_images(new_i, old_i)
    assert new_m[0] == old_m[0]
    assert new_m[1]["content"][:-1] == old_m[1]["content"][:-1]
    assert new_m[1]["content"][-1]["text"] == (
        old_m[1]["content"][-1]["text"][:-len(ANSWER_INSTRUCTION)] + core.STRICT_ANSWER_INSTRUCTION)
    # Quoted instruction inside the literal question must survive; no global replacement.
    assert question in new_m[1]["content"][-1]["text"]
    changed = {key for key in set(new_g) | set(old_g) if new_g.get(key) != old_g.get(key)}
    assert changed == {"question_prompt", "messages_sha256", "extraction_instruction_version"}
    assert new_g["extraction_instruction_version"] == "minimal_span_v1"
    assert new_g["messages_sha256"] == hashlib.sha256(
        json.dumps(new_m, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert new_g["previous_answer_in_prompt"] is False
    assert new_g["answer_page_privileged"] is True
    assert new_g["roi_annotation_used"] is False


@pytest.mark.parametrize("kind,expected_size", [("direct", (480, 512)), ("highres", (1024, 1024))])
def test_standalone_pixels_exactly_match_legacy_budget(source, kind, expected_size):
    old_m, old_i, old_g = legacy.build_answer_request(source, "Which date?", kind, CONFIG)
    new_m, new_i, new_g = core.make_request(source, "Which date?", f"new_{kind}", CONFIG)
    equal_images(new_i, old_i)
    assert len(new_i) == 1 and new_i[0].size == expected_size
    assert new_m[1]["content"][-1]["text"] == "Which date?\n" + core.STRICT_ANSWER_INSTRUCTION
    assert new_g["source_roi_pixels"] is None and new_g["roi_used"] is False
    assert new_g["image_rgb_sha256"] == old_g["image_rgb_sha256"]


@pytest.mark.parametrize("region", range(1, 10))
def test_legacy_selected_is_identical_to_committed_native_region(source, region):
    expected = legacy.build_answer_request(source, "Which address?", "native", CONFIG, region)
    selected = core.make_request(source, "Which address?", "legacy_selected", CONFIG, region)
    preflight = core.make_request(source, "Which address?", f"legacy_native_{region}", CONFIG)
    assert selected[0] == expected[0] == preflight[0]
    assert selected[2] == expected[2] == preflight[2]
    equal_images(selected[1], expected[1])
    equal_images(preflight[1], expected[1])
    assert core.preflight_key("legacy_selected", region) == f"legacy_native_{region}"


def test_selector_prompt_pixels_and_geometry_are_unchanged(source):
    expected = legacy.build_selector_request(source, "Which account?", CONFIG)
    actual = core.make_request(source, "Which account?", "legacy_selector", CONFIG)
    assert actual[0] == expected[0]
    assert actual[2] == expected[2]
    equal_images(actual[1], expected[1])
    assert core.STRICT_ANSWER_INSTRUCTION not in str(actual[0])


def test_manual_odd_dimension_boxes_and_corner_crop(source):
    # Width1025 gives floor-half512, remaining513, middle round(256.5)=256.
    # Height1027 gives floor-half513, remaining514, middle257.
    starts_x, starts_y = (0, 256, 513), (0, 257, 514)
    expected_boxes = [[x / 1025, y / 1027, (x + 512) / 1025, (y + 513) / 1027]
                      for y in starts_y for x in starts_x]
    assert core.normalized_boxes((1025, 1027)) == expected_boxes
    for box in expected_boxes:
        assert (box[2] - box[0]) * (box[3] - box[1]) <= .25
    # Independent explicit bottom-right geometry, without candidate helper reuse.
    _, images, geometry = core.make_request(source, "Which code?", "new_native_9", CONFIG)
    assert geometry["source_roi_pixels"] == [513, 514, 1025, 1027]
    expected = source.crop((513, 514, 1025, 1027)).resize((512, 512), Image.Resampling.BICUBIC)
    equal_images([images[1]], [expected])


def test_calls_are_fresh_and_do_not_mutate_input_or_legacy_requests(source):
    source_bytes = source.tobytes()
    cfg = copy.deepcopy(CONFIG)
    old = legacy.build_answer_request(source, "Question?", "native", cfg, 5)
    old_messages = copy.deepcopy(old[0])
    new = core.make_request(source, "Question?", "new_native_5", cfg)
    new[0][1]["content"][-1]["text"] = "Caller mutation"
    new[1][0].putpixel((0, 0), (255, 255, 255))
    assert source.tobytes() == source_bytes
    assert CONFIG == cfg
    assert old[0] == old_messages
    repeated = core.make_request(source, "Question?", "new_native_5", cfg)
    assert repeated[0][1]["content"][-1]["text"].endswith(core.STRICT_ANSWER_INSTRUCTION)
    assert repeated[1][0].tobytes() == old[1][0].tobytes()


def test_non_rgb_source_produces_same_rgb_pixels_without_source_mutation():
    source = Image.new("L", (1025, 1027), 123)
    before = source.tobytes()
    new = core.make_request(source, "Which text?", "new_native_1", CONFIG)
    old = legacy.build_answer_request(source, "Which text?", "native", CONFIG, 1)
    equal_images(new[1], old[1])
    assert source.mode == "L" and source.tobytes() == before


def test_action_plan_has_no_missing_or_duplicate_generation_or_preflight_actions():
    new = {"new_direct", "new_highres"} | {f"new_native_{i}" for i in range(1, 10)}
    assert set(core.NEW_ACTIONS) == new and len(core.NEW_ACTIONS) == 11
    assert set(core.GENERATION_ACTIONS) == new | {"legacy_selector", "legacy_selected"}
    assert len(core.GENERATION_ACTIONS) == 13
    assert set(core.PREFLIGHT_ACTIONS) == new | {"legacy_selector"} | {f"legacy_native_{i}" for i in range(1, 10)}
    assert len(core.PREFLIGHT_ACTIONS) == 21
    state = random.getstate()
    orders = [core.action_order(f"source:{i}") for i in range(8)]
    assert random.getstate() == state
    assert orders == [core.action_order(f"source:{i}") for i in range(8)]
    for order in orders:
        assert order[:2] == ["legacy_selector", "legacy_selected"]
        assert len(order) == len(set(order)) == 13 and set(order[2:]) == new
    assert len({tuple(order[2:]) for order in orders}) > 1


@pytest.mark.parametrize("region", [None, True, False, 0, 10, -1, "1", 1.0])
def test_uncommitted_or_noninteger_legacy_selected_region_rejected(source, region):
    with pytest.raises(ValueError):
        core.make_request(source, "Which code?", "legacy_selected", CONFIG, region)


@pytest.mark.parametrize("action", ["new_native_0", "new_native_10", "new_native_01", "new_native_+1",
                                  "legacy_native_0", "legacy_native_10", "legacy_native_01", "legacy_native_+1",
                                  "legacy_native_1 ", "new_degraded_1", "native_1", "", "legacy_native_"])
def test_only_canonical_planned_action_names_are_accepted(source, action):
    with pytest.raises(ValueError):
        core.make_request(source, "Which code?", action, CONFIG)


def test_unexpected_frozen_suffix_fails_closed(monkeypatch, source):
    original = core.build_answer_request
    def changed(*args, **kwargs):
        messages, images, geometry = original(*args, **kwargs)
        messages[-1]["content"][-1]["text"] += " A changed upstream suffix."
        return messages, images, geometry
    monkeypatch.setattr(core, "build_answer_request", changed)
    with pytest.raises(ValueError, match="suffix"):
        core.make_request(source, "Which text?", "new_direct", CONFIG)


def test_source_observation_boundary_still_is_the_frozen_five_field_type():
    assert core.Observation is legacy.Observation
    row = {"example_id": "example", "question": "Which text?", "image_path": "image.png",
           "image_sha256": "a" * 64, "source_cluster_id": "cluster"}
    assert core.Observation.from_manifest(row).question == "Which text?"
    with pytest.raises(ValueError):
        core.Observation.from_manifest({**row, "validated_primary_answers": ["Do not use this"]})
