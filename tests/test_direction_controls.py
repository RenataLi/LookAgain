"""CPU-only invariants for the frozen direction-control experiment."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from lookagain.actions import build_request

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("direction_controls_test_module", PROJECT / "experiments" / "direction_controls.py")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


@pytest.fixture
def config():
    return json.loads((PROJECT / "configs" / "direction_controls.json").read_text(encoding="utf-8"))


@pytest.fixture
def source():
    image = Image.new("RGB", (25, 19))
    image.putdata([(x * 10, y * 12, (x * 7 + y * 3) % 256) for y in range(19) for x in range(25)])
    return image


def pixels(image):
    return image.mode, image.size, image.tobytes()


def test_direct_and_named_requests_match_original_v1(config, source):
    for action in ["direct", *["named_" + region for region in experiment.REGIONS]]:
        original_action = "direct" if action == "direct" else action.replace("named_", "crop_")
        expected_messages, expected_images, _ = build_request(source, "Where is the object?", original_action, config, "ANSWER: left")
        messages, images, _ = experiment.build_condition(source, "Where is the object?", action, config, "ANSWER: left")
        assert messages == expected_messages
        assert [pixels(image) for image in images] == [pixels(image) for image in expected_images]


def test_true_crop_pixels_match_across_conditions_and_shift_in_correct_axes(config, source):
    views = {}
    expected_boxes = {"tl": (0, 0, 15, 12), "tr": (10, 0, 25, 12), "bl": (0, 7, 15, 19), "br": (10, 7, 25, 19)}
    for region in experiment.REGIONS:
        reference = None
        for condition in ("named", "neutral", "frame"):
            _, images, geometry = experiment.build_condition(source, "Where?", f"{condition}_{region}", config, "ANSWER: left")
            if reference is None:
                reference = [pixels(image) for image in images]
            assert [pixels(image) for image in images] == reference
            assert tuple(geometry["crop_box_pixels"]) == expected_boxes[region]
            assert geometry["actual_second_view_box"] == config["region_boxes"][region]
            expected_crop = source.crop(expected_boxes[region]).resize(images[1].size, Image.Resampling.BICUBIC)
            assert pixels(images[1]) == pixels(expected_crop)
            views[region] = images[1].getpixel((images[1].width // 2, images[1].height // 2))
    # Position-encoded channels catch flipped axes or a shifted crop origin.
    assert views["tr"][0] > views["tl"][0]
    assert views["br"][0] > views["bl"][0]
    assert views["bl"][1] > views["tl"][1]
    assert views["br"][1] > views["tr"][1]


def test_sham_is_exact_overview_copy_with_separate_claim_and_actual_box(config, source):
    for region in experiment.REGIONS:
        named_messages, _, _ = experiment.build_condition(source, "Where?", "named_" + region, config, "ANSWER: left")
        messages, images, geometry = experiment.build_condition(source, "Where?", "sham_" + region, config, "ANSWER: left")
        assert messages == named_messages
        assert pixels(images[0]) == pixels(images[1])
        assert images[0] is not images[1]
        assert geometry["image_rgb_sha256"][0] == geometry["image_rgb_sha256"][1]
        assert geometry["claimed_crop_box"] == config["region_boxes"][region]
        assert geometry["actual_second_view_box"] == [0.0, 0.0, 1.0, 1.0]
        assert geometry["additional_view_kind"] == "exact_overview_repeat"
        assert geometry["crop_box_pixels"] is None and geometry["crop_box_normalized"] is None
        original_pixel = images[0].getpixel((0, 0))
        images[1].putpixel((0, 0), (255, 255, 255))
        assert images[0].getpixel((0, 0)) == original_pixel


def test_requests_do_not_mutate_source_or_leak_prior_branch_state(config, source):
    original = pixels(source)
    previous_messages = None
    for action in experiment.ACTIONS[1:]:
        messages, _, geometry = experiment.build_condition(source, "Where?", action, config, "ANSWER: fresh direct")
        assert [row["content"] for row in messages if row["role"] == "assistant"] == ["ANSWER: fresh direct"]
        assert pixels(source) == original
        if geometry["condition"] == "neutral":
            assert geometry["claimed_crop_box"] is None
            assert all(location not in geometry["followup_prompt"] for location in experiment.LOCATIONS.values())
        if previous_messages is not None:
            assert previous_messages[2]["content"] == "mutated old request"
        messages[2]["content"] = "mutated old request"
        previous_messages = messages


@pytest.fixture
def fake_run(tmp_path, monkeypatch, config, source):
    """Run the real orchestration with an in-memory torch/API stand-in."""
    import lookagain.backend as backend_module
    import lookagain.data as data_module

    state = {"instances": 0, "direct_calls": 0, "followups": []}
    fake_cuda = SimpleNamespace(manual_seed_all=lambda _: None, get_device_name=lambda _: "CPU test stand-in",
                                synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                                max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
    fake_torch = SimpleNamespace(manual_seed=lambda _: None, cuda=fake_cuda,
                                 version=SimpleNamespace(cuda="test-runtime"),
                                 backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False, allow_tf32=False),
                                                          cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(experiment.importlib.metadata, "version", lambda _: "test-version")

    class FakeBackend:
        def __init__(self, *_):
            state["instances"] += 1

        def generate(self, messages, images, max_new_tokens, answer_prefix=False):
            assert answer_prefix is True
            first_question = messages[1]["content"][-1]["text"]
            if first_question.startswith("What color is the image?"):
                response = "ANSWER: gray"
            elif not any(row["role"] == "assistant" for row in messages):
                state["direct_calls"] += 1
                response = "ANSWER: original direct"
            else:
                initial = [row["content"] for row in messages if row["role"] == "assistant"]
                state["followups"].append(initial)
                response = f"ANSWER: branch {len(state['followups'])}"
            return {"response": response, "generation_truncated": False, "generated_tokens": 3}

    monkeypatch.setattr(backend_module, "QwenBackend", FakeBackend)
    manifest = tmp_path / "manifest.jsonl"
    image_path = tmp_path / "synthetic.png"
    source.save(image_path)
    rows = [{"example_id": "one", "image_id": "one-image", "image_path": image_path.name,
             "question": "Synthetic state check?", "answer": "unused synthetic reference"}]
    manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    monkeypatch.setattr(data_module, "validate_manifest", lambda _: copy.deepcopy(rows))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "lookagain-model.json").write_text(json.dumps({"model_id": config["model_id"], "revision": config["protocol"]["model_revision"]}), encoding="utf-8")
    config["base_visual_tokens"] = config["crop_visual_tokens"] = 16
    config["protocol"]["manifest_sha256"] = experiment.sha256_file(manifest)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return SimpleNamespace(manifest=manifest, model_dir=model_dir, config_path=config_path,
                           config=config, output=tmp_path / "run", state=state, torch=fake_torch)


def test_real_orchestration_uses_one_fresh_direct_for_every_independent_branch(fake_run):
    f = fake_run
    experiment.run(f.manifest, f.model_dir, f.config_path, f.output)
    assert f.state["direct_calls"] == 1
    assert f.state["followups"] == [["ANSWER: original direct"]] * 16
    completed = experiment.read_records(f.output / "records.jsonl")
    assert len(completed) == 17
    assert {key[1] for key in completed} == set(experiment.ACTIONS)
    # The varying branch outputs must never become later branches' initial state.
    assert len({row["response"] for row in completed.values()}) == 17


def test_resume_skips_completed_calls_and_rejects_runtime_or_config_changes(fake_run):
    f = fake_run
    experiment.run(f.manifest, f.model_dir, f.config_path, f.output)
    experiment.run(f.manifest, f.model_dir, f.config_path, f.output)
    assert f.state["instances"] == 1 and f.state["direct_calls"] == 1
    f.torch.version.cuda = "changed-runtime"
    with pytest.raises(ValueError, match="Run identity changed"):
        experiment.run(f.manifest, f.model_dir, f.config_path, f.output)
    f.torch.version.cuda = "test-runtime"
    f.config["seed"] += 1
    f.config_path.write_text(json.dumps(f.config), encoding="utf-8")
    with pytest.raises(ValueError, match="Run identity changed"):
        experiment.run(f.manifest, f.model_dir, f.config_path, f.output)
    assert f.state["instances"] == 1


def test_duplicate_or_failed_records_are_rejected(tmp_path):
    path = tmp_path / "records.jsonl"
    row = {"example_id": "one", "action": "direct", "status": "ok"}
    path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate or unsuccessful"):
        experiment.read_records(path)
    row["status"] = "error"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate or unsuccessful"):
        experiment.read_records(path)


def test_config_rejects_broken_control_text_or_action_set(config):
    experiment.validate_config(config)
    mutations = [lambda c: c["condition_prompts"].update(sham="Different text"),
                 lambda c: c["condition_prompts"].update(neutral="Look at {location}"),
                 lambda c: c["actions"].pop()]
    for mutate in mutations:
        changed = copy.deepcopy(config)
        mutate(changed)
        with pytest.raises(ValueError):
            experiment.validate_config(changed)
