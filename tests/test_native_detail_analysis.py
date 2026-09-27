"""Independent aggregation fixtures and strict provenance rejection tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
SPEC = importlib.util.spec_from_file_location("native_analysis_test_module", PROJECT / "experiments" / "analyze_native_detail.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def synthetic_groups():
    groups = []
    for index in range(2):
        group = {}
        for action in analysis.ACTIONS:
            right = (index == 1) if action in ("direct", "repeat") else (
                index == 0 if action == "highres" else action == "native_tl")
            group[action] = {
                "correct": bool(right), "conservative_text_em": float(right),
                "official_em": float(right), "official_f1": float(right), "anls": float(right),
                "parse_valid": True, "generation_truncated": False,
                "elapsed_s": 2.0 if action == "direct" else 5.0 if action == "highres" else 1.0,
                "peak_memory_gib": 8.0 if action == "direct" else 12.0 if action == "highres" else 6.0,
                "peak_reserved_gib": 14.0 if action == "direct" else 16.0 if action == "highres" else 9.0,
                "input_tokens": 100 if action == "direct" else 400 if action == "highres" else 200,
                "visual_tokens": 64 if action == "direct" else 256 if action == "highres" else 128,
                "generated_tokens": 1 if action == "direct" else 3 if action == "highres" else 2,
            }
        groups.append(group)
    return groups


@pytest.fixture
def small_config():
    return {"analysis": {"bootstrap_samples": 100, "bootstrap_seed": 20260925,
                         "primary_metric": "conservative_text_em", "practical_delta": .02}}


def test_primary_keeps_four_quadrants_in_source_cluster(small_config):
    result = analysis.summarize(synthetic_groups(), small_config)
    primary = result["contrasts"]["native_minus_degraded"]["metrics"]["conservative_text_em"]
    # Each source has exactly 1/4 native correct. A presentation bootstrap would have nonzero variance.
    assert primary["difference"] == .25
    assert primary["ci95"] == [.25, .25]
    assert primary["n_source_reports"] == 2
    assert result["conditions"]["native"]["n_presentations"] == 8
    assert result["conditions"]["native"]["metrics"]["conservative_text_em"] == .25


def test_paired_contrast_preserves_negative_effects_and_pairs():
    draws = [[0, 0], [0, 1], [1, 0], [1, 1]] * 25
    score = analysis.paired_contrast([1.0, 0.0], [0.0, .5], draws)
    assert score["difference"] == .25 and score["ci95"] == [-.5, 1.0]
    zero = analysis.paired_contrast([.1, .9], [.1, .9], draws)
    assert zero["ci95"] == [0, 0]
    with pytest.raises(ValueError, match="Unpaired"):
        analysis.paired_contrast([1], [1, 0], draws)


def test_fixes_harms_have_conditional_and_distinct_source_denominators():
    row = analysis.changes_vs_direct(synthetic_groups(), analysis.CONDITIONS["native"])
    assert row["n_presentations"] == 8
    assert row["fixes"] == 1 and row["harms"] == 3
    assert row["direct_wrong_presentations"] == row["direct_correct_presentations"] == 4
    assert row["fix_rate_among_direct_wrong"] == .25
    assert row["harm_rate_among_direct_correct"] == .75
    assert row["distinct_sources_with_any_fix"] == row["distinct_sources_with_any_harm"] == 1
    assert row["diagnostic_anls_changes"] == {
        "positive": 1, "negative": 3, "zero": 4,
        "mean_positive_change": 1, "mean_negative_change": -1, "mean_change": -.25}


def test_standalone_costs_followup_sum_memory_max_and_all_tokens(small_config):
    result = analysis.summarize(synthetic_groups(), small_config)
    high = result["actions"]["highres"]
    native = result["actions"]["native_tl"]
    assert high["full_path_latency_s"]["mean"] == 5
    assert native["full_path_latency_s"]["mean"] == 3
    assert native["sequential_peak_allocated_gib"]["mean"] == 8
    assert native["sequential_peak_reserved_gib"]["mean"] == 14
    assert native["tokens"]["input_tokens"]["full_policy"]["mean"] == 300
    assert native["tokens"]["visual_tokens"]["full_policy"]["mean"] == 192
    assert native["tokens"]["generated_tokens"]["full_policy"]["mean"] == 3
    assert high["tokens"]["input_tokens"]["full_policy"]["mean"] == 400


def test_oracles_match_candidate_capacity_exclude_highres_and_show_harms(small_config):
    result = analysis.summarize(synthetic_groups(), small_config)
    oracle = result["privileged_oracles"]
    native = oracle["oracles"]["native"]
    degraded = oracle["oracles"]["degraded"]
    assert len(native["candidate_actions"]) == len(degraded["candidate_actions"]) == 6
    assert len(oracle["oracles"]["union"]["candidate_actions"]) == 10
    assert all("highres" not in row["candidate_actions"] for row in oracle["oracles"].values())
    assert native["metrics"]["conservative_text_em"] == 1
    assert degraded["metrics"]["conservative_text_em"] == .5
    assert oracle["matched_native_minus_degraded"]["conservative_text_em"]["difference"] == .5
    assert native["optimistic_primary_selected_full_path_latency_s"]["mean"] == 2.5
    assert native["primary_selected_action_counts"]["direct"] == 1
    assert native["primary_selected_action_counts"]["native_tl"] == 1
    assert result["gate"]["proceed_headroom_rule_met"]
    assert result["gate"]["revision_flags"]["fixed_native_below_direct"]
    assert result["gate"]["decision"] == "revise_before_larger_study"


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def strict_run(tmp_path, small_config):
    """Synthetic pages exist only inside pytest's work directory, never report outputs."""
    config = {**small_config, "actions": list(analysis.ACTIONS), "model_id": "synthetic/model",
              "dtype": "bfloat16", "attention": "sdpa", "do_sample": False,
              "short_answer_prefix": "ANSWER:", "answer_max_tokens": 64,
              "base_visual_tokens": 1024, "crop_visual_tokens": 1024, "highres_visual_tokens": 4096,
              "protocol": {"rules_locked_at_utc": "2026-09-25T00:00:00Z", "main_examples": 2,
                           "model_revision": "synthetic-revision"}}
    sources, records = [], []
    for index in range(2):
        image_path = tmp_path / f"synthetic-{index}.png"
        image = Image.new("RGB", (2048, 1024), (index * 20, 80, 100))
        image.save(image_path)
        source = {"example_id": str(index), "image_id": "page-" + str(index), "source_id": "report-" + str(index),
                  "question": "Synthetic question?", "answer": "5 units", "image_path": image_path.name,
                  "image_sha256": analysis.sha256_file(image_path)}
        sources.append(source)
        response = "ANSWER: 5 units"
        for action in analysis.ACTIONS:
            _, images, geometry = analysis.build_request(image, source["question"], action, config, response)
            grids = [[1, im.height // 16, im.width // 16] for im in images]
            visual = sum(t * h * w // 4 for t, h, w in grids)
            scores = analysis.score_response(response, source["answer"])
            scores.update(analysis.score_official(scores["predicted_answer"], source["answer"]))
            records.append({**geometry, **scores, "example_id": source["example_id"], "image_id": source["image_id"],
                "source_id": source["source_id"], "question": source["question"], "target_answer": source["answer"],
                "source_image_sha256": source["image_sha256"], "action": action, "status": "ok",
                "response": response, "raw_continuation": " 5 units", "answer_prefix_prefilled": True,
                "generated_tokens": 3, "input_tokens": visual + 50, "visual_tokens": visual,
                "image_grid_thw": grids, "generation_truncated": False,
                "elapsed_s": 1.0, "peak_memory_gib": 8.0, "peak_reserved_gib": 9.0})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8")
    config["protocol"]["main_manifest_sha256"] = analysis.sha256_file(manifest)
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    runtime = {"gpu": "CPU synthetic fixture", "cuda": "none", "packages": {}, "python": "test", "platform": "test"}
    identity = {"config": config, "manifest_sha256": analysis.sha256_file(manifest), "code_sha256": analysis.experiment_digest(),
                "model": {"model_id": config["model_id"], "revision": "synthetic-revision"},
                "runtime": runtime, "role": "main", "example_ids": [r["example_id"] for r in sources]}
    metadata = {**identity, **runtime, "fingerprint": analysis.canonical_hash(identity),
                "base_code_sha256": analysis.code_digest(), "created_utc": "2026-09-25T01:00:00Z",
                "experiment": "native_detail_v3", "development_only": True, "cost_definition": "synthetic costs"}
    write_json(tmp_path / "run.json", metadata)
    def save_rows(rows):
        record_path = tmp_path / "records.jsonl"
        record_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        write_json(tmp_path / "completed.json", {"records": len(rows), "records_sha256": analysis.sha256_file(record_path), "elapsed_s": 1.0})
    save_rows(records)
    return tmp_path, manifest, config_path, records, save_rows


def test_complete_synthetic_analysis_reconstructs_inputs_and_reports(strict_run):
    run, manifest, config_path, _, _ = strict_run
    summary = analysis.analyze(run, manifest, config_path)
    assert summary["status"] == "complete"
    assert summary["coverage"]["records"] == 22
    assert summary["coverage"]["source_reports"] == 2
    assert summary["integrity"]["all_view_pixels_and_chat_reconstructed"] is True
    assert "not deployable controllers" in analysis.markdown(summary)


def test_wholly_absent_example_cannot_survive_as_easy_subset(strict_run):
    run, manifest, config_path, rows, save_rows = strict_run
    save_rows([row for row in rows if row["example_id"] != "1"])
    with pytest.raises(ValueError, match="Expected exactly 22"):
        analysis.analyze(run, manifest, config_path)


@pytest.mark.parametrize("field,value", [("messages_sha256", "bad"), ("image_rgb_sha256", ["bad", "bad"]),
                                         ("official_em", .123), ("input_tokens", 9999)])
def test_forged_scores_pixels_chat_or_pair_tokens_fail_closed(strict_run, field, value):
    run, manifest, config_path, rows, save_rows = strict_run
    modified = copy.deepcopy(rows)
    row = next(row for row in modified if row["action"] == "native_tl")
    row[field] = value
    save_rows(modified)  # Rehashing records cannot conceal a semantic/provenance violation.
    with pytest.raises(ValueError):
        analysis.analyze(run, manifest, config_path)


def test_bad_identity_is_rejected_before_aggregation(strict_run):
    run, manifest, config_path, _, _ = strict_run
    meta = analysis.read_json(run / "run.json")
    meta["role"] = "smoke"
    write_json(run / "run.json", meta)
    with pytest.raises(ValueError, match="fingerprint"):
        analysis.analyze(run, manifest, config_path)


def bind_quality_mask(strict_run, *, full_check=True):
    run, manifest, config_path, _, _ = strict_run
    mask = {
        "frozen_before_main_inference": True,
        "review_is_before_main_inference": True,
        "model_predictions_or_outcomes_accessed_by_reviewers": False,
        "main_manifest_sha256": analysis.sha256_file(manifest),
        "locked_at_utc": "2026-09-25T00:30:00Z",
        "categories_by_example_id": {"0": "clear", "1": "ambiguous", "smoke-only": "label_error"},
        "mask": {
            "0": {"category": "clear", "cohort": "main", "full_page_visual_check": False},
            "1": {"category": "ambiguous", "cohort": "main", "full_page_visual_check": full_check, "reason": "Synthetic source-only ambiguity"},
            "smoke-only": {"category": "label_error", "cohort": "smoke", "full_page_visual_check": True}},
        "secondary_excluded_main_ids": ["1"],
    }
    for row in analysis.read_jsonl(manifest):
        mask["mask"][row["example_id"]].update(source_id=row["source_id"], source_image_sha256=row["image_sha256"],
                                              question=row["question"], reference_answer=row["answer"])
    path = run / "mask.json"
    write_json(path, mask)
    config = analysis.read_json(config_path)
    config["analysis"]["quality_mask_sha256"] = analysis.sha256_file(path)
    config["protocol"]["source_audit_locked_at_utc"] = "2026-09-25T00:30:00Z"
    write_json(config_path, config)
    meta = analysis.read_json(run / "run.json")
    meta["config"] = config
    meta["fingerprint"] = analysis.canonical_hash({key: meta[key] for key in analysis.IDENTITY_KEYS})
    write_json(run / "run.json", meta)
    return path


def test_frozen_quality_mask_is_secondary_and_never_changes_primary(strict_run):
    run, manifest, config_path, _, _ = strict_run
    mask = bind_quality_mask(strict_run)
    result = analysis.analyze(run, manifest, config_path, mask)
    assert result["coverage"]["source_reports"] == 2
    assert result["conditions"]["native"]["n_presentations"] == 8
    secondary = result["source_quality_sensitivity"]
    assert secondary["retained_n_source_reports"] == 1
    assert secondary["conditions"]["native"]["n_presentations"] == 4
    assert secondary["excluded_example_ids"] == ["1"]
    assert secondary["gate_applied"] is False and "gate" not in secondary
    assert secondary["bootstrap"]["n_source_reports"] == 1
    assert "Secondary source-quality sensitivity" in analysis.markdown(result)


def test_pinned_mask_required_and_exclusions_need_full_page_audit(strict_run):
    run, manifest, config_path, _, _ = strict_run
    mask = bind_quality_mask(strict_run, full_check=False)
    with pytest.raises(ValueError, match="supply --quality-mask"):
        analysis.analyze(run, manifest, config_path)
    with pytest.raises(ValueError, match="full-page"):
        analysis.analyze(run, manifest, config_path, mask)
