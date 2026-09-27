"""Independent analytic fixtures for paired v2 estimates and report integrity."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_module("analyze_direction_controls")
runner = load_module("direction_controls")
gallery = load_module("render_direction_gallery")


def make_group(index, direct, outcomes):
    group = {"direct": {"correct": direct}}
    for condition, values in outcomes.items():
        group.update({f"{condition}_{region}": {"correct": bool(value)}
                      for region, value in zip(analysis.REGIONS, values)})
    for action, row in group.items():
        row.update(example_id=str(index), image_id=f"image-{index}",
                   question="Is it on the left?", target_answer="left",
                   predicted_answer="left" if row["correct"] else "right", parse_valid=True,
                   generation_truncated=False, generated_tokens=2,
                   elapsed_s=10.0 * index if action == "direct" else float(index),
                   peak_memory_gib=(7.0 if index == 1 else 11.0) if action == "direct" else (3.0 if index == 1 else 13.0))
    return group


@pytest.fixture
def groups():
    return [make_group(1, False, {"named": [1, 0, 0, 0], "neutral": [1, 1, 1, 0],
                                  "sham": [0, 0, 0, 0], "frame": [1, 1, 1, 1]}),
            make_group(2, True, {"named": [1, 1, 1, 1], "neutral": [1, 0, 0, 0],
                                 "sham": [1, 1, 0, 0], "frame": [0, 0, 0, 0]})]


def test_primary_contrast_signs_uniform_quadrants_and_whole_image_bootstrap(groups):
    result = analysis.primary_contrasts(groups)
    # Image differences are +50 and -75 pp, each with four correlated views.
    # Resampling two whole images yields only -75, -12.5 or +50, with
    # probabilities 1/4, 1/2, 1/4. A presentation-level bootstrap is narrower.
    neutral = result["neutral_minus_named"]
    assert neutral["n_images"] == 2
    assert neutral["delta_expected_accuracy_pp"] == -12.5
    assert neutral["delta_ci_95_pp"] == [-75.0, 50.0]
    assert result["named_minus_sham"]["delta_expected_accuracy_pp"] == 37.5
    assert result["frame_minus_named"]["delta_expected_accuracy_pp"] == -12.5
    assert analysis.transition_metrics(groups, ["named_" + region for region in analysis.REGIONS])["expected_accuracy"] == 0.625


def test_identical_per_image_conditions_have_zero_paired_uncertainty():
    groups = [make_group(1, False, {condition: [0] * 4 for condition in analysis.CONDITIONS}),
              make_group(2, True, {condition: [1] * 4 for condition in analysis.CONDITIONS})]
    for metric in analysis.primary_contrasts(groups).values():
        assert metric["delta_expected_accuracy_pp"] == 0
        assert metric["delta_ci_95_pp"] == [0.0, 0.0]


def test_fix_harm_denominators_count_presentations_but_not_independent_images(groups):
    metric = analysis.transition_metrics(groups, ["neutral_" + region for region in analysis.REGIONS])
    assert metric["n_images"] == 2 and metric["n_presentations"] == 8
    assert metric["fix_presentations"] == metric["harm_presentations"] == 3
    assert metric["fix_fraction_all_presentations"] == 3 / 8
    assert metric["harm_fraction_all_presentations"] == 3 / 8
    assert metric["fix_rate_denominator_presentations"] == metric["harm_rate_denominator_presentations"] == 4
    assert metric["fix_rate_given_direct_wrong"] == metric["harm_rate_given_direct_correct"] == 3 / 4
    assert metric["unique_images_with_any_fix"] == metric["unique_images_with_any_harm"] == 1
    empty = analysis.transition_metrics([], ["named_tl"])
    assert empty["expected_accuracy"] is None and empty["fix_rate_given_direct_wrong"] is None


def test_directional_diagnostics_do_not_treat_invalid_or_nondirectional_answers_as_left_or_right(groups):
    group = copy.deepcopy(groups[0])
    group["direct"].update(predicted_answer="left", target_answer="right")
    for region, prediction in zip(analysis.REGIONS, ("Left.", "right", "no", "right")):
        group["neutral_" + region]["predicted_answer"] = prediction
    group["neutral_br"]["parse_valid"] = False
    result = analysis.direction_metrics([group], "neutral")
    assert result["directional_answer_count"] == 2
    assert result["nondirectional_or_invalid_answer_count"] == 2
    assert result["left_to_right_denominator"] == 2
    assert result["direct_left_to_followup_right"] == 1
    assert result["target_right_directional_prediction_denominator"] == 2
    assert result["target_right_to_predicted_left_rate"] == 0.5
    assert result["text_cue_alignment_count"] is None
    assert result["region_side_alignment_rate_all_presentations"] == 0.5


def test_sham_horizontal_comparisons_are_height_matched(groups):
    group = copy.deepcopy(groups[0])
    for region, prediction in zip(analysis.REGIONS, ("left", "right", "no", "right")):
        group["sham_" + region]["predicted_answer"] = prediction
    result = analysis.sham_cue_sensitivity([group])
    assert result["height_matched_left_right_pairs"] == 2
    assert result["unique_images_with_any_direction_category_change"] == 1
    cross = {(row["left_cue_answer"], row["right_cue_answer"]): row["pair_count"]
             for row in result["left_cue_answer_by_right_cue_answer"]}
    assert cross[("left", "right")] == 1 and cross[("other_or_invalid", "right")] == 1
    assert sum(cross.values()) == 2


def test_analysis_and_gallery_charge_direct_once_and_take_maximum_memory(groups, monkeypatch):
    metadata = {"config": {"analysis": {"minimum_complete_examples_for_primary_report": 2}},
                "fingerprint": "0" * 64, "manifest_sha256": "1" * 64,
                "code_sha256": "2" * 64, "model": {"model_id": "synthetic"}}
    monkeypatch.setattr(analysis, "validate_run", lambda *args, **kwargs: (metadata, {}, groups, 2))
    monkeypatch.setattr(analysis, "sha256_file", lambda _: "3" * 64)
    summary = analysis.analyze(Path("unused-run"), Path("unused-manifest"))
    cost = summary["cost_by_action_descriptive"]
    assert cost["direct"]["full_path_latency_s"]["mean"] == 15
    assert cost["named_tl"]["full_path_latency_s"]["mean"] == 16.5
    assert cost["named_tl"]["sequential_peak_memory_gib"]["mean"] == 10
    assert cost["named_tl"]["sequential_peak_memory_gib"]["max"] == 13
    cell = gallery.result_cell(groups[0]["named_tl"], groups[0]["direct"])
    assert "Total 11.000 s" in cell and "Extra 1.000 s" in cell


@pytest.fixture
def valid_run(tmp_path):
    config = json.loads((PROJECT / "configs" / "direction_controls.json").read_text(encoding="utf-8"))
    config["base_visual_tokens"] = config["crop_visual_tokens"] = 16
    config["analysis"]["minimum_complete_examples_for_primary_report"] = 1
    source = Image.new("RGB", (25, 19))
    source.putdata([(x * 10, y * 12, 0) for y in range(19) for x in range(25)])
    image_path = tmp_path / "source.png"
    source.save(image_path)
    item = {"example_id": "one", "image_id": "one-image", "image_path": image_path.name,
            "image_sha256": analysis.sha256_file(image_path), "question": "Is it left?", "answer": "left"}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(item) + "\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    metadata = {"config": config, "manifest_sha256": analysis.sha256_file(manifest), "code_sha256": "0" * 64,
                "model": {"model_id": config["model_id"]}, "runtime": {}, "example_ids": ["one"]}
    metadata["fingerprint"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    (run / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    rows = []
    for action in analysis.ACTIONS:
        _, _, geometry = runner.build_condition(source, item["question"], action, config, "ANSWER: left")
        rows.append({"example_id": "one", "image_id": "one-image", "action": action, "question": item["question"],
                     "target_answer": "left", "status": "ok", "correct": True, "parse_valid": True,
                     "predicted_answer": "left", "generation_truncated": False, "generated_tokens": 2,
                     "elapsed_s": 1.0, "peak_memory_gib": 2.0, **geometry})
    (run / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return run, manifest, rows


def test_actual_validation_requires_all_seventeen_outcomes(valid_run):
    run, manifest, rows = valid_run
    assert len(analysis.validate_run(run, manifest)[2]) == 1
    (run / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete planned run"):
        analysis.validate_run(run, manifest)


def test_actual_validation_rejects_sham_pixel_mismatch(valid_run):
    run, manifest, rows = valid_run
    next(row for row in rows if row["action"] == "sham_tr")["image_rgb_sha256"][1] = "f" * 64
    (run / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Sham is not an exact overview repeat"):
        analysis.validate_run(run, manifest)
