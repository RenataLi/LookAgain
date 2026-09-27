"""V5 comparisons isolate detail while keeping regional content and chat fixed."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import evidence_availability_core as core
import evidence_availability as runner
from native_detail_core import build_request as old_request


@pytest.fixture(scope="module")
def source():
    return Image.effect_noise((1654, 2339), 80).convert("RGB")


def test_identical_regional_prompts_and_fixed_native_pixels(source):
    config = dict(base_visual_tokens=1024, crop_visual_tokens=1024, highres_visual_tokens=4096)
    native_hashes = []
    for budget in core.BUDGETS:
        native = core.build_request(source, "Which year?", f"native_{budget}", config, [131, 512, 899, 800])
        degraded = core.build_request(source, "Which year?", f"degraded_{budget}", config, [131, 512, 899, 800])
        direct = core.build_request(source, "Which year?", f"direct_{budget}", config, [131, 512, 899, 800])
        assert native[0] == degraded[0]
        assert [m["role"] for m in native[0]] == ["system", "user"]
        assert native[2]["image_rgb_sha256"][0] == degraded[2]["image_rgb_sha256"][0] == direct[2]["image_rgb_sha256"][0]
        assert native[2]["image_rgb_sha256"][1] != degraded[2]["image_rgb_sha256"][1]
        assert native[2]["additional_size"] == degraded[2]["additional_size"]
        assert native[2]["additional_size"][0] * native[2]["additional_size"][1] >= 65536
        native_hashes.append(native[2]["image_rgb_sha256"][1])
    assert len(set(native_hashes)) == 1


@pytest.mark.parametrize("action,old_action", [("direct_1024", "direct"), ("highres", "highres")])
def test_overview_controls_preserve_old_requests(source, action, old_action):
    config = dict(base_visual_tokens=1024, crop_visual_tokens=1024, highres_visual_tokens=4096)
    new = core.build_request(source, "Q?", action, config, [0, 0, 256, 256])
    old = old_request(source, "Q?", old_action, config)
    assert new[0] == old[0]
    assert new[2]["image_rgb_sha256"] == old[2]["image_rgb_sha256"]


@pytest.mark.parametrize("roi", [[0, 0, 255, 256], [-1, 0, 300, 300], [0, 0, 256, 3000], [0., 0, 256, 256], [True, 0, 256, 256]])
def test_rejects_unsafe_or_undersized_roi(source, roi):
    with pytest.raises(ValueError):
        core.build_request(source, "Q?", "native_256", {"crop_visual_tokens": 1024}, roi)


def test_pair_checker_detects_measurement_and_pixel_changes():
    base = dict(messages_sha256="m", overview_size=[416,608], additional_size=[256,256], source_roi_pixels=[0,0,256,256],
                projected_overview_roi_pixels=[0,0,40,60], image_grid_thw=[[1,26,38],[1,16,16]], input_tokens=400,
                visual_tokens=311, image_rgb_sha256=["overview", "native"])
    rows = {("q", "native_256"): base, ("q", "degraded_256"): {**base, "image_rgb_sha256":["overview","degraded"]}}
    runner.check_pair(rows, "q", 256)
    bad = copy.deepcopy(rows)
    bad["q", "degraded_256"]["input_tokens"] += 1
    with pytest.raises(ValueError, match="input_tokens"):
        runner.check_pair(bad, "q", 256)
    rows["q", "native_512"] = {**base, "image_rgb_sha256":["bigger", "different"]}
    with pytest.raises(ValueError, match="vary across"):
        runner.check_pair(rows, "q", 256)


def test_no_target_parameter_and_no_hidden_history(source):
    messages, _, geo = core.build_request(source, "Q_SENTINEL?", "native_256", {"crop_visual_tokens":1024}, [0,0,256,256])
    assert messages[-1]["content"][-1]["text"] == core.FRESH_DESCRIPTION + "\nQ_SENTINEL?\n" + core.ANSWER_INSTRUCTION
    assert geo["roi_localizer"] == "reference_annotation_privileged"
    assert geo["previous_answer_in_prompt"] is False


def test_resume_recomputes_saved_identity_instead_of_trusting_fingerprint():
    identity = {"config":{"seed":20260926}, "runtime":{"dtype":"bfloat16"}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    saved = {**copy.deepcopy(identity), "fingerprint":fingerprint, "created_utc":"metadata"}
    runner.validate_saved_identity(saved, identity, fingerprint)
    saved["config"]["seed"] = 1
    with pytest.raises(ValueError, match="identity changed"):
        runner.validate_saved_identity(saved, identity, fingerprint)
