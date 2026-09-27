"""History ablations must change conversation content, never matched pixels."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import history_context_core as core
import history_context as runner
from native_detail_core import build_request as v3_request


@pytest.fixture(scope="module")
def config():
    return dict(base_visual_tokens=1024, crop_visual_tokens=1024, highres_visual_tokens=4096)


@pytest.fixture(scope="module")
def source():
    # Asymmetric texture makes native versus projected-overview differences observable.
    return Image.effect_noise((1654, 2339), 80).convert("RGB")


@pytest.mark.parametrize("base_action", ["direct", "highres", "repeat", *(
    f"{f}_{r}" for f in core.FIDELITIES for r in core.REGIONS)])
def test_actual_exactly_preserves_v3(base_action, source, config):
    action = base_action if base_action in ("direct", "highres", "repeat") else "actual_" + base_action
    old, old_images, old_geo = v3_request(source, "What is the rate?", base_action, config, "ANSWER: 17.4%")
    new, new_images, new_geo = core.build_request(source, "What is the rate?", action, config, "ANSWER: 17.4%")
    assert old == new
    assert [i.tobytes() for i in old_images] == [i.tobytes() for i in new_images]
    assert all(new_geo[key] == value for key, value in old_geo.items())


@pytest.mark.parametrize("region", core.REGIONS)
def test_pixels_match_across_history_and_messages_match_across_fidelity(region, source, config):
    records = {}
    for history in core.HISTORIES:
        for fidelity in core.FIDELITIES:
            action = f"{history}_{fidelity}_{region}"
            messages, images, geometry = core.build_request(source, "Which year?", action, config, "ANSWER: 2099_UNIQUE")
            records[history, fidelity] = geometry
            assert len(images) == 2
            if history == "fresh":
                assert [m["role"] for m in messages] == ["system", "user"]
                assert [part["type"] for part in messages[1]["content"]] == ["image", "image", "text"]
                assert geometry["initial_answer_sha256"] is None
            else:
                assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
            if history != "actual":
                assert "2099_UNIQUE" not in json.dumps(messages)
                assert not geometry["previous_answer_in_prompt"]
            if history == "placeholder":
                assert messages[2]["content"] == core.PLACEHOLDER
                assert geometry["initial_answer_sha256"] == hashlib.sha256(core.PLACEHOLDER.encode()).hexdigest()
        assert records[history, "native"]["messages_sha256"] == records[history, "degraded"]["messages_sha256"]
        assert records[history, "native"]["additional_size"] == records[history, "degraded"]["additional_size"]
        assert records[history, "native"]["image_rgb_sha256"][1] != records[history, "degraded"]["image_rgb_sha256"][1]
    for fidelity in core.FIDELITIES:
        assert len({tuple(records[h, fidelity]["image_rgb_sha256"]) for h in core.HISTORIES}) == 1


def test_branches_do_not_use_stale_answers(source, config):
    for action in ("fresh_native_tl", "placeholder_degraded_br"):
        left = core.build_request(source, "Which year?", action, config, "ANSWER: 1999")
        right = core.build_request(source, "Which year?", action, config, "ANSWER: 2025")
        assert left[0] == right[0]
        assert left[2] == right[2]
    with pytest.raises(ValueError, match="requires"):
        core.build_request(source, "Q?", "actual_native_tl", config)
    with pytest.raises(ValueError, match="Unknown"):
        core.build_request(source, "Q?", "bogus", config)


def test_pair_checker_rejects_cross_history_pixel_change():
    record = dict(messages_sha256="m", followup_prompt="p", overview_size=[32, 32], additional_size=[32, 32],
                  image_grid_thw=[[1, 2, 2], [1, 2, 2]], input_tokens=100, visual_tokens=2,
                  image_rgb_sha256=["base", "native"])
    rows = {("q", "actual_native_tl"): record, ("q", "fresh_native_tl"): {**record, "image_rgb_sha256": ["base", "wrong"]}}
    with pytest.raises(ValueError, match="across histories"):
        runner.check_pair(rows, "q", "tl")
    rows = {("q", "actual_native_tl"): record, ("q", "actual_degraded_tl"): {**record, "input_tokens": 99}}
    with pytest.raises(ValueError, match="input_tokens"):
        runner.check_pair(rows, "q", "tl")
