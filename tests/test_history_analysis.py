"""CPU-only independent estimand fixtures and completed-record integrity tests."""
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
SPEC = importlib.util.spec_from_file_location("history_analysis_tests", PROJECT / "experiments/analyze_history_context.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def synthetic_groups():
    groups = []
    for index in range(2):
        group = {}
        for action in analysis.ACTIONS:
            correct = index == 1 if action in ("direct", "repeat") else action == "highres" or action in ("placeholder_native_tl", "fresh_native_tl", "fresh_native_tr")
            response = "ANSWER: right" if correct else "ANSWER: wrong"
            group[action] = {"correct": bool(correct), **{m: float(correct) for m in analysis.METRICS},
                "parse_valid": True, "generation_truncated": False, "response": response,
                "predicted_answer": "right" if correct else "wrong",
                "elapsed_s": 2. if action == "direct" else 5. if action == "highres" else 1.,
                "peak_memory_gib": 8. if action == "direct" else 12. if action == "highres" else 6.,
                "peak_reserved_gib": 14. if action == "direct" else 16. if action == "highres" else 9.,
                "input_tokens": 100 if action == "direct" else 400 if action == "highres" else 200,
                "visual_tokens": 64 if action == "direct" else 256 if action == "highres" else 128,
                "generated_tokens": 1 if action == "direct" else 3 if action == "highres" else 2}
        groups.append(group)
    return groups


@pytest.fixture
def small_config():
    return {"analysis": {"bootstrap_samples": 100, "bootstrap_seed": 20260925,
        "primary_metric": "conservative_text_em", "candidate_gate": {"detail_effect_threshold": .02}}}


def test_interactions_are_formed_within_report_and_keep_quadrants_together(small_config):
    result = analysis.summarize(synthetic_groups(), small_config)
    for name, expected in ((analysis.PRIMARY, .25), ("fresh_minus_actual_detail_interaction", .5),
                           ("fresh_minus_placeholder_detail_interaction", .25)):
        stat = result["contrasts"][name]["metrics"]["conservative_text_em"]
        assert stat["difference"] == expected
        assert stat["ci95"] == [expected, expected]
        assert stat["n_source_reports"] == 2
    assert result["conditions"]["placeholder_native"]["n_presentations"] == 8
    assert result["conditions"]["placeholder_native"]["metrics"]["conservative_text_em"] == .25
    assert result["detail_effects"]["fresh"]["conservative_text_em"]["difference"] == .5
    for name in ("repeat_minus_direct", "actual_degraded_minus_direct", "fresh_degraded_minus_highres"):
        assert name in result["contrasts"]


def test_paired_bootstrap_keeps_negative_changes_and_pairing():
    draws = [[0, 0], [0, 1], [1, 0], [1, 1]] * 25
    stat = analysis.paired_contrast([1., 0.], [0., .5], draws)
    assert stat["difference"] == .25 and stat["ci95"] == [-.5, 1.]
    assert analysis.paired_contrast([.1, .9], [.1, .9], draws)["ci95"] == [0., 0.]
    with pytest.raises(ValueError, match="Unpaired"):
        analysis.paired_contrast([1], [1, 0], draws)


def test_dual_costs_preserve_direct_dependency_and_sequential_memory(small_config):
    result = analysis.summarize(synthetic_groups(), small_config)
    for name in ("fresh_native_tl", "placeholder_native_tl"):
        row = result["actions"][name]
        assert row["decision_state_latency_s"]["mean"] == 3
        assert row["standalone_latency_s"]["mean"] == 1
        assert row["decision_state_peak_allocated_gib"]["mean"] == 8
        assert row["standalone_peak_allocated_gib"]["mean"] == 6
        assert row["tokens"]["input_tokens"]["decision_state"]["mean"] == 300
        assert row["tokens"]["input_tokens"]["standalone"]["mean"] == 200
    for name in ("actual_native_tl", "repeat"):
        assert result["actions"][name]["standalone_latency_s"]["mean"] == 3
    assert result["actions"]["highres"]["decision_state_latency_s"]["mean"] == 5
    assert result["conditions"]["fresh_native"]["decision_state_latency_s"]["mean"] == 3  # Not four branches.


def test_raw_invalid_equality_is_counted_but_not_parsed_equality():
    groups = synthetic_groups()
    for group in groups:
        for action in ("direct", "fresh_native_tl"):
            group[action].update(response="ANSWER:", predicted_answer=None, parse_valid=False)
    result = analysis.response_persistence(groups, ["fresh_native_tl"])
    assert result["counts"] == {"raw_response_exact": 2, "parsed_answer_exact": 0, "normalized_answer": 0}
    assert result["fractions"]["raw_response_exact"] == 1
    assert result["n_presentations"] == result["invalid_pairs_retained_in_denominator"] == 2
    paired = analysis.paired_response_agreement(groups, ["direct"], ["fresh_native_tl"])
    assert paired["raw_response_exact"] == 2 and paired["normalized_answer_equal"] == 0


def test_persistence_normalizes_case_and_whitespace_without_stripping_units():
    groups = synthetic_groups()[:1]
    groups[0]["direct"].update(response="ANSWER: USD  5", predicted_answer="USD  5")
    groups[0]["fresh_native_tl"].update(response="ANSWER: usd 5", predicted_answer="usd 5")
    result = analysis.response_persistence(groups, ["fresh_native_tl"])
    assert result["counts"] == {"raw_response_exact": 0, "parsed_answer_exact": 0, "normalized_answer": 1}
    groups[0]["fresh_native_tl"]["predicted_answer"] = "5"
    assert analysis.response_persistence(groups, ["fresh_native_tl"])["counts"]["normalized_answer"] == 0


def test_oracles_have_equal_five_action_capacity_and_gate_is_not_training(small_config):
    result = analysis.summarize(synthetic_groups(), small_config)
    for row in result["privileged_oracles"]["oracles"].values():
        assert len(row["candidate_actions"]) == 5
        assert "direct" in row["candidate_actions"]
        assert "highres" not in row["candidate_actions"] and "repeat" not in row["candidate_actions"]
    native = result["privileged_oracles"]["oracles"]["placeholder_native"]
    assert native["metrics"]["conservative_text_em"] == 1
    assert native["primary_selected_action_counts"]["direct"] == 1
    assert native["optimistic_selected_decision_state_latency_s"]["mean"] == 2.5
    gate = result["candidate_gate"]
    assert gate["histories"]["fresh"]["candidate_rule_met"]
    assert not gate["histories"]["placeholder"]["candidate_rule_met"]  # Gain positive but below direct quality.
    assert gate["automatic_controller_training"] is False
    assert len(result["cross_history_response_agreement"]) == 6


def test_fixes_and_harms_keep_correct_and_incorrect_direct_denominators():
    changes = analysis.changes_vs_direct(synthetic_groups(), analysis.CONDITIONS["placeholder_native"])
    assert changes["fixes"] == 1 and changes["harms"] == 3
    assert changes["direct_wrong_presentations"] == changes["direct_correct_presentations"] == 4
    assert changes["fix_rate_among_direct_wrong"] == .25 and changes["harm_rate_among_direct_correct"] == .75


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def strict_run(tmp_path):
    """Exercise validation with isolated synthetic inputs."""
    config = copy.deepcopy(analysis.read_json(PROJECT / "configs/history_context.json"))
    # Synthetic records bind their own protocol copy and SHA-256.
    protocol_path = tmp_path / "synthetic_protocol.md"
    protocol_path.write_bytes((PROJECT / config["protocol"]["document"]).read_bytes())
    config["protocol"]["document"] = str(protocol_path.resolve())
    config["protocol"]["document_sha256"] = analysis.sha256_file(protocol_path)
    config["analysis"].update(bootstrap_samples=100, expected_secondary_sources=1)
    config["protocol"]["main_examples"] = 2
    sources, records = [], []
    response = "ANSWER: 5 units"
    for index in range(2):
        image = Image.new("RGB", (2048, 1024), (index * 20, 80, 100))
        image_path = tmp_path / f"synthetic-{index}.png"
        image.save(image_path)
        source = {"example_id": str(index), "image_id": f"page-{index}", "source_id": f"report-{index}",
            "question": "Synthetic question?", "answer": "5 units", "image_path": image_path.name,
            "image_sha256": analysis.sha256_file(image_path)}
        sources.append(source)
        for action in analysis.ACTIONS:
            _, images, geometry = analysis.build_request(image, source["question"], action, config, response)
            grids = [[1, im.height // 16, im.width // 16] for im in images]
            visual = sum(t * h * w // 4 for t, h, w in grids)
            scores = analysis.score_response(response, source["answer"])
            scores.update(analysis.score_official(scores["predicted_answer"], source["answer"]))
            records.append({**geometry, **scores, "example_id": source["example_id"], "image_id": source["image_id"],
                "source_id": source["source_id"], "question": source["question"], "target_answer": source["answer"],
                "source_image_sha256": source["image_sha256"], "action": action, "status": "ok",
                "observed_direct_response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "response": response, "raw_continuation": " 5 units", "answer_prefix_prefilled": True,
                "generated_tokens": 3, "input_tokens": visual + 50, "visual_tokens": visual,
                "image_grid_thw": grids, "generation_truncated": False,
                "elapsed_s": 1., "peak_memory_gib": 8., "peak_reserved_gib": 9.})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8")
    config["protocol"]["main_manifest_sha256"] = analysis.sha256_file(manifest)
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    runtime = {"gpu": "CPU synthetic fixture", "cuda": "none", "packages": {}, "python": "test", "platform": "test"}
    identity = {"config": config, "manifest_sha256": analysis.sha256_file(manifest), "code_sha256": analysis.experiment_digest(),
        "model": {"model_id": config["model_id"], "revision": config["protocol"]["model_revision"]},
        "runtime": runtime, "role": "main", "example_ids": [r["example_id"] for r in sources]}
    metadata = {**identity, **runtime, "fingerprint": analysis.canonical_hash(identity),
        "base_code_sha256": analysis.code_digest(), "created_utc": "2026-09-26T00:00:00Z",
        "experiment": "history_context_v4", "development_only": True}
    write_json(tmp_path / "run.json", metadata)
    def save_rows(rows):
        record_path = tmp_path / "records.jsonl"
        record_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        write_json(tmp_path / "completed.json", {"records": len(rows), "records_sha256": analysis.sha256_file(record_path),
            "elapsed_s": 1., "finished_utc": "2026-09-26T01:00:00Z"})
    save_rows(records)
    return tmp_path, manifest, config_path, records, save_rows


def test_complete_synthetic_records_reconstruct_all_requests(strict_run):
    run, manifest, config, _, _ = strict_run
    _, sources, groups, _ = analysis.validate_run(run, manifest, config)
    assert len(sources) == 2 and len(groups) == 2
    assert all(len(group) == 27 for group in groups)


@pytest.mark.parametrize("change", ["document_bytes", "recorded_digest"])
def test_synthetic_protocol_binding_rejects_changed_bytes_or_digest(strict_run, change):
    run, manifest, config_path, _, _ = strict_run
    analysis.validate_run(run, manifest, config_path)
    config = analysis.read_json(config_path)
    if change == "document_bytes":
        protocol = Path(config["protocol"]["document"])
        protocol.write_bytes(protocol.read_bytes() + b"\nChanged synthetic protocol.\n")
    else:
        config["protocol"]["document_sha256"] = "0" * 64
        write_json(config_path, config)
        metadata = analysis.read_json(run / "run.json")
        metadata["config"] = config
        metadata["fingerprint"] = analysis.canonical_hash({k: metadata[k] for k in analysis.IDENTITY_KEYS})
        write_json(run / "run.json", metadata)
    with pytest.raises(ValueError, match="Protocol document changed"):
        analysis.validate_run(run, manifest, config_path)


@pytest.mark.parametrize("field,value", [("observed_direct_response_sha256", "wrong"),
    ("inserted_history_text", "changed"), ("messages_sha256", "wrong"),
    ("image_rgb_sha256", ["wrong", "wrong"]), ("official_em", .123), ("input_tokens", 9999)])
def test_rehashed_records_cannot_hide_content_corruption(strict_run, field, value):
    run, manifest, config, records, save_rows = strict_run
    changed = copy.deepcopy(records)
    next(row for row in changed if row["action"] == "placeholder_native_tl")[field] = value
    save_rows(changed)
    with pytest.raises(ValueError):
        analysis.validate_run(run, manifest, config)


@pytest.mark.parametrize("change", ["missing_source", "duplicate", "smoke_identity", "fingerprint", "completion_hash"])
def test_partial_or_wrong_identity_panels_are_rejected(strict_run, change):
    run, manifest, config, rows, save_rows = strict_run
    if change == "missing_source":
        save_rows([row for row in rows if row["example_id"] == "0"])
    elif change == "duplicate":
        save_rows([*rows[:-1], rows[0]])
    elif change in ("smoke_identity", "fingerprint"):
        metadata = analysis.read_json(run / "run.json")
        metadata["role"] = "smoke"
        if change == "smoke_identity":
            metadata["fingerprint"] = analysis.canonical_hash({k: metadata[k] for k in analysis.IDENTITY_KEYS})
        write_json(run / "run.json", metadata)
    else:
        marker = analysis.read_json(run / "completed.json")
        marker["records_sha256"] = "bad"
        write_json(run / "completed.json", marker)
    with pytest.raises(ValueError):
        analysis.validate_run(run, manifest, config)


def test_recovered_complete_marker_preserves_unavailable_wall_time(strict_run):
    run, manifest, config, _, _ = strict_run
    marker = analysis.read_json(run / "completed.json")
    marker.update(elapsed_s=None, note="Completion marker recovered after all records were durable; total wall time unavailable.")
    write_json(run / "completed.json", marker)
    assert analysis.validate_run(run, manifest, config)[3]["elapsed_s"] is None
    marker.pop("note")
    write_json(run / "completed.json", marker)
    with pytest.raises(ValueError, match="Missing study elapsed"):
        analysis.validate_run(run, manifest, config)


def bind_mask(strict_run):
    run, manifest, config_path, _, _ = strict_run
    sources = analysis.read_jsonl(manifest)
    mask = {"frozen_before_main_inference": True, "review_is_before_main_inference": True,
        "model_predictions_or_outcomes_accessed_by_reviewers": False,
        "main_manifest_sha256": analysis.sha256_file(manifest),
        **{k: "2026-09-25T20:00:00Z" for k in ("locked_at_utc", "created_utc", "frozen_at_utc")},
        "categories_by_example_id": {"0": "mapping_only", "1": "ambiguous"},
        "secondary_excluded_main_ids": ["1"], "mask": {}}
    for row in sources:
        mask["mask"][row["example_id"]] = {"category": mask["categories_by_example_id"][row["example_id"]],
            "cohort": "main", "full_page_visual_check": row["example_id"] == "1", "reason": "Synthetic source flag",
            "source_id": row["source_id"], "source_image_sha256": row["image_sha256"],
            "question": row["question"], "reference_answer": row["answer"]}
    path = run / "mask.json"
    write_json(path, mask)
    config = analysis.read_json(config_path)
    config["analysis"]["quality_mask_sha256"] = analysis.sha256_file(path)
    write_json(config_path, config)
    metadata = analysis.read_json(run / "run.json")
    metadata["config"] = config
    metadata["fingerprint"] = analysis.canonical_hash({k: metadata[k] for k in analysis.IDENTITY_KEYS})
    write_json(run / "run.json", metadata)
    return path


def test_frozen_mask_leaves_primary_unchanged_and_mapping_only_retained(strict_run, monkeypatch):
    run, manifest, config, _, _ = strict_run
    mask = bind_mask(strict_run)
    monkeypatch.setattr(analysis, "archived_v3_agreement", lambda *args: {"role": "Synthetic archive stub; separately validated in real run", "actions": {}})
    result = analysis.analyze(run, manifest, mask, config)
    assert result["coverage"]["source_reports"] == 2
    assert result["conditions"]["actual_native"]["n_presentations"] == 8
    secondary = result["source_quality_sensitivity"]
    assert secondary["retained_n_source_reports"] == 1 and secondary["excluded_example_ids"] == ["1"]
    assert secondary["per_source_example_ids"] == ["0"]
    assert secondary["candidate_gate_applied"] is False and "candidate_gate" not in secondary
    assert result["bindings"]["records_sha256"] == analysis.sha256_file(run / "records.jsonl")
    assert analysis.PRIMARY in analysis.markdown(result)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value", [("full_page_visual_check", False), ("question", "different")])
def test_mask_source_flags_require_bound_full_page_evidence(strict_run, field, value):
    run, manifest, config, _, _ = strict_run
    path = bind_mask(strict_run)
    mask = analysis.read_json(path)
    mask["mask"]["1"][field] = value
    write_json(path, mask)
    metadata = analysis.read_json(run / "run.json")
    metadata["config"]["analysis"]["quality_mask_sha256"] = analysis.sha256_file(path)
    with pytest.raises(ValueError):
        analysis.validate_quality_mask(path, metadata, analysis.read_jsonl(manifest))


def test_archived_v3_hash_mismatch_fails_before_reading_records(strict_run):
    run, _, _, _, _ = strict_run
    metadata = analysis.read_json(run / "run.json")
    with pytest.raises(ValueError, match="Archived v3 binding"):
        analysis.archived_v3_agreement(run, metadata, [])
