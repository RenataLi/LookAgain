"""CPU-only v5 paired estimand, cost and rehashed-content integrity tests."""
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
SPEC = importlib.util.spec_from_file_location("evidence_analysis_tests", PROJECT / "experiments/analyze_evidence_availability.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
import evidence_availability as runner
from evidence_availability_core import ANSWER_INSTRUCTION, FRESH_DESCRIPTION


def config_for_statistics():
    return {"analysis": {"primary_metric": "official_em", "bootstrap_samples": 100,
            "bootstrap_seed": 20260926, "candidate_gate": {"detail_effect_threshold": .02}}}


def synthetic_groups():
    groups = []
    for index in range(2):
        group = {}
        for action in analysis.ACTIONS:
            kind, budget = analysis.parse_action(action)
            correct = (kind == "direct" and index == 1) or action == "highres" or action == "native_256" or (budget == 1024 and kind in ("native", "degraded") and index == 0)
            score = float(correct)
            group[action] = {"official_em": score, "conservative_text_em": 0., "official_f1": score, "anls": score,
                "parse_valid": True, "generation_truncated": False,
                "response": "ANSWER: correct" if correct else "ANSWER: wrong", "predicted_answer": "correct" if correct else "wrong",
                "elapsed_s": {256: 2., 512: 3., 1024: 4.}.get(budget, 9.) if kind in ("direct", "highres") else 1.,
                "peak_memory_gib": {256: 8., 512: 12., 1024: 16.}.get(budget, 20.) if kind in ("direct", "highres") else 6.,
                "peak_reserved_gib": 22., "input_tokens": 100, "visual_tokens": 64, "generated_tokens": 3}
        groups.append(group)
    return groups


def test_primary_and_interaction_are_paired_by_report_not_presentations():
    result = analysis.summarize(synthetic_groups(), config_for_statistics())
    primary = result["contrasts"][analysis.PRIMARY]["metrics"]["official_em"]
    interaction = result["contrasts"][analysis.INTERACTION]["metrics"]["official_em"]
    assert primary["difference"] == interaction["difference"] == 1.
    assert primary["ci95"] == interaction["ci95"] == [1., 1.]
    assert primary["n_source_reports"] == 2
    assert result["detail_effects"]["1024"]["official_em"]["difference"] == 0
    assert result["detail_effects"]["256"]["conservative_text_em"]["difference"] == 0
    assert result["bootstrap"]["unit"] == "original_source_report"
    assert result["bootstrap"]["retains_all_budgets_and_fidelities"] is True


def test_costs_add_only_matching_budget_and_sequential_memory_uses_maximum():
    group = synthetic_groups()[0]
    assert analysis.cost(group, "native_512") == 1.
    assert analysis.cost(group, "native_512", mode="decision_state") == 4.
    assert analysis.cost(group, "degraded_1024", mode="decision_state") == 5.
    assert analysis.cost(group, "native_512", "peak_memory_gib", "decision_state") == 12.
    assert analysis.cost(group, "direct_1024", mode="decision_state") == 4.
    assert analysis.cost(group, "highres", mode="decision_state") == 9.
    assert analysis.cost(group, "native_256", "input_tokens", "decision_state") == 200
    with pytest.raises(ValueError, match="Unknown cost"):
        analysis.cost(group, "native_256", mode="invented")


def test_interaction_keeps_correlated_budget_effects_in_same_resampled_report():
    groups = synthetic_groups()
    for index, group in enumerate(groups):
        for budget in (256, 1024):
            group[f"native_{budget}"]["official_em"] = float(index == 0)
            group[f"degraded_{budget}"]["official_em"] = 0.
    result = analysis.summarize(groups, config_for_statistics())
    # Each budget effect varies across reports, but within-report differences
    # between effects are all zero. Independently resampling arms would fail.
    assert result["detail_effects"]["256"]["official_em"]["ci95"] == [0., 1.]
    interaction = result["contrasts"][analysis.INTERACTION]["metrics"]["official_em"]
    assert interaction["difference"] == 0.
    assert interaction["ci95"] == [0., 0.]


def test_quantitative_gate_never_implies_controller_ready_or_semantic_pass():
    result = analysis.summarize(synthetic_groups(), config_for_statistics())
    gate = result["candidate_gate"]
    assert gate["quantitative_rule_met"] is True
    assert gate["semantic_audit_status"] == "pending_separate_artifact"
    assert gate["controller_ready"] is gate["automatic_controller_training"] is False
    privileged = result["privileged_localization"]
    assert privileged["annotation_location_is_privileged_information"] is True
    assert privileged["target_text_in_language_prompt"] is False
    assert privileged["new_unseen_report_confirmation"] is False


def test_transition_denominators_and_undefined_conditional_rate():
    result = analysis.transitions(synthetic_groups(), "native_256", "degraded_256", "official_em")
    assert result["fixes"] == result["negative_wrong"] == 2
    assert result["fix_rate_among_negative_wrong"] == 1
    assert result["negative_correct"] == 0
    assert result["harm_rate_among_negative_correct"] is None
    assert result["correct_to_correct"] == result["wrong_to_wrong"] == 0


def test_audit_list_keeps_equal_score_answer_changes_and_invalid_pairs():
    groups = synthetic_groups()
    groups[0]["native_256"].update(predicted_answer="(1,234)", response="ANSWER: (1,234)", official_em=1.)
    groups[0]["degraded_256"].update(predicted_answer="1234", response="ANSWER: 1234", official_em=1.)
    groups[1]["native_256"].update(predicted_answer=None, parse_valid=False, response="ANSWER:")
    report = analysis.pair_diagnostics(groups, "native_256", "degraded_256")
    assert report["parsed_answer_changed"] == 2
    assert report["metric_equal"]["official_em"] == 1
    assert report["both_parse_valid"] == 1


def test_audit_includes_raw_differences_with_two_invalid_parses():
    groups = synthetic_groups()[:1]
    groups[0]["native_256"].update(predicted_answer=None, parse_valid=False, response="ANSWER:")
    groups[0]["degraded_256"].update(predicted_answer=None, parse_valid=False, response="ANSWER:   ")
    result = analysis.pair_diagnostics(groups, "native_256", "degraded_256")
    assert result["parsed_answer_changed"] == 0
    assert result["raw_response_changed_with_invalid_parse_source_indices"] == [0]
    assert result["semantic_audit_required_source_indices"] == [0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_nested_configuration_is_rejected(bad):
    with pytest.raises(ValueError, match="Nonfinite"):
        analysis.finite_tree({"outer": [{"parameter": bad}]})


def test_missing_action_cannot_produce_summary():
    groups = synthetic_groups()
    groups[0].pop("highres")
    with pytest.raises(ValueError, match="Complete ten-action"):
        analysis.summarize(groups, config_for_statistics())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def strict_run(tmp_path):
    """Real reconstruction/validators, fake sources and no GPU or real outcomes."""
    config = {**config_for_statistics(), "actions": analysis.ACTIONS, "overview_budgets": list(analysis.BUDGETS),
        "model_id": "synthetic/model", "dtype": "bfloat16", "attention": "sdpa", "do_sample": False,
        "short_answer_prefix": "ANSWER:", "answer_max_tokens": 64, "crop_visual_tokens": 1024, "highres_visual_tokens": 4096,
        "prompts": {"answer": ANSWER_INSTRUCTION, "fresh_context": FRESH_DESCRIPTION}}
    protocol_file = tmp_path / "synthetic_protocol.md"
    protocol_file.write_text("Synthetic unit fixture, never a research result.\n", encoding="utf-8")
    config["protocol"] = {"inference_code_sha256": runner.experiment_digest(), "document": str(protocol_file),
        "document_sha256": analysis.sha256_file(protocol_file), "model_revision": "synthetic-revision",
        "rules_locked_at_utc": "2026-09-26T00:00:00Z", "main_execution_locked_at_utc": "2026-09-26T00:00:00Z",
        "main_examples": 2, "smoke_examples": 2}
    sources, records, audits, decisions = [], [], [], []
    for index in range(2):
        image = Image.new("RGB", (2048, 1024), (index * 30, 80, 100))
        image_path = tmp_path / f"synthetic-{index}.png"
        image.save(image_path)
        source = {"example_id": str(index), "image_id": f"page-{index}", "source_id": f"report-{index}",
            "question": "Synthetic number?", "answer": "5", "image_path": image_path.name,
            "image_sha256": analysis.sha256_file(image_path), "source_pdf_sha256": "a" * 64,
            "pixel_dimensions": [2048, 1024], "roi_pixels": [0, 0, 256, 256], "roi_privileged": True,
            "source_quality_category": "none", "source_manifest_sha256": analysis.SOURCE_PINS["source_main_manifest_sha256"]}
        audit = {"example_id": str(index), "source_id": source["source_id"], "cohort": "main", "question": source["question"],
            "reference_answer": "5", "source_quality_category": "none", "source_manifest_sha256": source["source_manifest_sha256"],
            "source_image_sha256": source["image_sha256"], "source_pdf_sha256": source["source_pdf_sha256"], "source_size": [2048, 1024],
            "pdf_geometry": {"page_count": 1, "mediabox": [0, 0, 2048, 1024], "cropbox": [0, 0, 2048, 1024], "rotation": 0, "user_unit": 1},
            "evidence": {"selected_text": "5", "unsafe_boundary_cut": False, "selected_words": [{"word": "5", "bbox": [96, 96, 144, 112]}], "ocr_page_bbox": [0, 0, 2048, 1024]},
            "roi_pixels": source["roi_pixels"], "privileged_annotation_roi": True, "model_outcomes_used": False,
            "construction": {"raw_word_bbox_ocr": [96, 96, 144, 112], "raw_word_bbox_pixels": [96, 96, 144, 112],
                "median_word_height_pixels": 16., "context_margin_pixels": 32., "minimum_roi_side_pixels": 256,
                "roi_page_area_fraction": .03125, "transform": {"scale_xy": [1., 1.], "offset_xy": [0., 0.],
                    "pdf_to_render_scale_xy": [1., 1.], "ocr_origin_xy": [0, 0]}}}
        source["roi_provenance_sha256"] = runner.canonical_hash(audit)
        audits.append(audit)
        sources.append(source)
        decisions.append({"example_id": str(index), "source_id": source["source_id"], "cohort": "main", "eligible": True,
                          "source_quality_category": "none", "roi_pixels": source["roi_pixels"], "roi_provenance_sha256": source["roi_provenance_sha256"]})
        response = "ANSWER: 5"
        for action in analysis.ACTIONS:
            _, images, geometry = analysis.build_request(image, source["question"], action, config, source["roi_pixels"])
            grids = [[1, im.height//16, im.width//16] for im in images]
            visual = sum(t*h*w//4 for t, h, w in grids)
            scores = analysis.score_response(response, source["answer"])
            scores.update(analysis.score_official(scores["predicted_answer"], source["answer"]))
            records.append({**geometry, **scores, "example_id": source["example_id"], "image_id": source["image_id"],
                "source_id": source["source_id"], "question": source["question"], "target_answer": source["answer"],
                "source_image_sha256": source["image_sha256"], "roi_provenance_sha256": source["roi_provenance_sha256"],
                "action": action, "status": "ok", "response": response, "raw_continuation": " 5", "answer_prefix_prefilled": True,
                "generated_tokens": 2, "input_tokens": visual+50, "visual_tokens": visual, "image_grid_thw": grids,
                "generation_truncated": False, "elapsed_s": 1., "peak_memory_gib": 8., "peak_reserved_gib": 9.})
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, sources)
    write_jsonl(tmp_path / "roi_audit.jsonl", audits)
    prep = {"created_utc": "2026-09-26T00:00:00Z", "stage": "evidence_availability_development",
        "selection_uses_model_outputs": False, "same_prior_development_reports": True, "held_out_claim": False,
        "privileged_annotation_roi": True, "source_images_copied_byte_identically": True,
        "preparation_script_sha256": analysis.sha256_file(PROJECT / "experiments/prepare_evidence_roi.py"), "pins": analysis.SOURCE_PINS,
        "main_manifest_sha256": analysis.sha256_file(manifest), "roi_audit_sha256": analysis.sha256_file(tmp_path / "roi_audit.jsonl"),
        "counts": {"main": {"eligible_sources": 2}}, "decisions": decisions}
    write_json(tmp_path / "selection_metadata.json", prep)
    config["protocol"].update(main_manifest_sha256=analysis.sha256_file(manifest),
        preparation_metadata_sha256=analysis.sha256_file(tmp_path / "selection_metadata.json"), roi_audit_sha256=prep["roi_audit_sha256"])
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    runtime = {"gpu": "CPU synthetic fixture", "cuda": "none", "packages": {}, "python": "test", "platform": "test"}
    identity = {"config": config, "manifest_sha256": analysis.sha256_file(manifest), "code_sha256": runner.experiment_digest(),
        "model": {"model_id": config["model_id"], "revision": config["protocol"]["model_revision"]},
        "runtime": runtime, "role": "main", "example_ids": [s["example_id"] for s in sources]}
    metadata = {**identity, **runtime, "fingerprint": analysis.canonical_hash(identity), "base_code_sha256": analysis.code_digest(),
        "created_utc": "2026-09-26T01:00:00Z", "experiment": "evidence_availability_v5", "development_only": True, "label_privileged_localizer": True}
    write_json(tmp_path / "run.json", metadata)
    def save_rows(rows):
        path = tmp_path / "records.jsonl"
        write_jsonl(path, rows)
        write_json(tmp_path / "completed.json", {"records": len(rows), "records_sha256": analysis.sha256_file(path),
            "elapsed_s": 1., "sum_measured_action_elapsed_s": sum(r["elapsed_s"] for r in rows), "finished_utc": "2026-09-26T02:00:00Z"})
    save_rows(records)
    return tmp_path, manifest, config_path, records, save_rows


def test_complete_fake_panel_passes_reconstruction_and_report_generation(strict_run):
    run, manifest, config, _, _ = strict_run
    metadata, sources, groups, completion = analysis.validate_run(run, manifest, config)
    assert len(sources) == len(groups) == 2
    assert all(len(g) == 10 for g in groups)
    result = analysis.analyze(run, manifest, config)
    assert result["integrity"]["preparation_and_annotation_geometry_validated"] is True
    assert result["coverage"]["records"] == 20
    assert "privileged" in analysis.markdown(result).lower()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value", [("question", "leaked target"), ("target_answer", "6"),
    ("roi_provenance_sha256", "wrong"), ("source_roi_pixels", [0, 0, 320, 320]),
    ("messages_sha256", "wrong"), ("image_rgb_sha256", ["wrong", "wrong"]),
    ("official_em", .5), ("conservative_text_em", 0.), ("input_tokens", 99999),
    ("history_mode", "actual"), ("previous_answer_in_prompt", True), ("generation_truncated", 1),
    ("peak_memory_gib", -1), ("elapsed_s", float("nan"))])
def test_rehashed_records_do_not_hide_content_or_measurement_tampering(strict_run, field, value):
    run, manifest, config, records, save_rows = strict_run
    rows = copy.deepcopy(records)
    next(r for r in rows if r["action"] == "native_256")[field] = value
    save_rows(rows)
    with pytest.raises(ValueError):
        analysis.validate_run(run, manifest, config)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed", "missing_truncation"])
def test_partial_failed_duplicate_and_missing_flag_rejected(strict_run, mutation):
    run, manifest, config, records, save_rows = strict_run
    rows = copy.deepcopy(records)
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "failed": rows[-1]["status"] = "failed"
    else: rows[-1].pop("generation_truncated")
    save_rows(rows)
    with pytest.raises(ValueError):
        analysis.validate_run(run, manifest, config)


def test_preparation_hash_and_original_image_bytes_are_required(strict_run):
    run, manifest, config, _, _ = strict_run
    prep = run / "selection_metadata.json"
    original = prep.read_bytes()
    prep.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="Preparation binding"):
        analysis.validate_run(run, manifest, config)
    prep.write_bytes(original)
    image = run / "synthetic-0.png"
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="image hash"):
        analysis.validate_run(run, manifest, config)


def test_geometry_reconstruction_rejects_wrong_transform_even_if_roi_unmodified(strict_run):
    run, _, _, _, _ = strict_run
    audit = analysis.read_jsonl(run / "roi_audit.jsonl")[0]
    analysis.validate_roi_geometry(audit)
    audit["construction"]["transform"]["scale_xy"][0] = 2
    with pytest.raises(ValueError, match="transform"):
        analysis.validate_roi_geometry(audit)


def test_smoke_flag_cannot_accept_main_or_partial_main(strict_run):
    run, manifest, config, _, _ = strict_run
    with pytest.raises(ValueError, match="Wrong panel role"):
        analysis.analyze(run, manifest, config, allow_smoke=True)


def test_recovered_marker_requires_explicit_unavailable_elapsed_note(strict_run):
    run, manifest, config, _, _ = strict_run
    marker = analysis.read_json(run / "completed.json")
    marker.update(elapsed_s=None, note="Completion marker recovered after all records were durable; total wall time unavailable.")
    write_json(run / "completed.json", marker)
    assert analysis.validate_run(run, manifest, config)[3]["elapsed_s"] is None
    marker.pop("note")
    write_json(run / "completed.json", marker)
    with pytest.raises(ValueError, match="Missing study elapsed"):
        analysis.validate_run(run, manifest, config)
