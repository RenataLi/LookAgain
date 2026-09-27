"""Independent numerical/control-flow checks for completed-only DUDE analysis."""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
spec = importlib.util.spec_from_file_location("dude_analysis_tests",
                                             PROJECT / "experiments/analyze_dude_replication.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture(vectors=None, n=4):
    if vectors is None:
        vectors = {"direct_256": [0, 1, 0, 1], "native_256": [1, 0, 1, 0],
                   "degraded_256": [0, 1, 1, 0], "highres": [1, 1, 1, 1]}
    n = len(vectors["direct_256"])
    rows, records = [], {}
    for i in range(n):
        eid, cid = f"example-{i}", f"source-{i}"
        rows.append({"example_id": eid, "source_cluster_id": cid,
                     "question": "What is the text?", "original_answers": ["Gold"],
                     "original_answer_variants": [], "validated_primary_answers": ["Gold"]})
        for a in analysis.ACTIONS:
            score = float(vectors[a][i])
            prediction = "Gold" if score else "Wrong"
            records[eid, a] = {
                "example_id": eid, "source_cluster_id": cid, "action": a, "status": "ok",
                "primary_em": score, "official_anls": score, "parse_valid": True,
                "predicted_answer": prediction, "normalized_prediction": prediction.casefold(),
                "response": "ANSWER: " + prediction, "generation_truncated": False,
                "elapsed_s": {"direct_256": 7., "native_256": 2.,
                              "degraded_256": 3., "highres": 11.}[a] + i,
                "peak_memory_gib": {"direct_256": 12., "native_256": 8.,
                                    "degraded_256": 9., "highres": 14.}[a],
                "peak_reserved_gib": 16., "input_tokens": 200, "visual_tokens": 128,
                "generated_tokens": 3, "processor_observer_elapsed_s": .1,
            }
    return rows, records


def all_same(n, score):
    return fixture({a: [score] * n for a in analysis.ACTIONS})


def config():
    return json.loads((PROJECT / "configs/dude_replication.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("b,c,p", [(0, 0, 1.), (1, 0, 1.), (5, 0, .0625),
                                  (6, 0, .03125), (0, 6, .03125),
                                  (8, 2, .109375), (10, 10, 1.), (660, 0, 2. ** -659)])
def test_exact_mcnemar_known_values(b, c, p):
    result = analysis.exact_mcnemar(b, c)
    assert result["p_two_sided"] == pytest.approx(p, rel=1e-14, abs=0)
    assert result["positive_direction_rejection"] == (b > c and p <= .05)


def test_exact_test_matches_probability_ordering_enumeration_not_normal_approximation():
    # Independent two-sided binomial definition: sum probabilities no greater
    # than the observed point mass under p=.5, using exact rational arithmetic.
    for discordant in range(36):
        for b in range(discordant + 1):
            masses = [Fraction(math.comb(discordant, k), 2 ** discordant)
                      for k in range(discordant + 1)]
            expected = sum(p for p in masses if p <= masses[b])
            actual = analysis.exact_mcnemar(b, discordant-b)["p_two_sided"]
            assert actual == float(expected)


@pytest.mark.parametrize("b,c", [(-1, 0), (True, 0), (0., 1), (1, float("nan"))])
def test_exact_counts_reject_invalid_types(b, c):
    with pytest.raises(ValueError):
        analysis.exact_mcnemar(b, c)


def test_paired_bootstrap_matches_independent_python_reference():
    pos, neg = [1, 1, 0, 1, 0], [0, 1, 1, 0, 0]
    draws = analysis.bootstrap_draws(5, 10000, analysis.SEED)
    rng = random.Random(analysis.SEED)
    reference = []
    for _ in range(10000):
        chosen = [rng.randrange(5) for _ in range(5)]
        reference.append(sum(pos[i]-neg[i] for i in chosen) / 5)
    reference.sort()
    def percentile(q):
        rank = q * (len(reference)-1)
        lo, hi = math.floor(rank), math.ceil(rank)
        return reference[lo] + (reference[hi]-reference[lo]) * (rank-lo)
    result = analysis.paired_interval(pos, neg, draws)
    assert result["difference"] == .2
    assert result["ci95"] == pytest.approx([percentile(.025), percentile(.975)])


def test_cluster_pairing_retains_covariance():
    common = [0, 1, 0, 1, 1, 0]
    result = analysis.paired_interval(common, common, analysis.bootstrap_draws(6, 1000))
    assert result["difference"] == 0.
    assert result["ci95"] == [0., 0.]
    assert result["degenerate_empirical_interval"] is True


def test_missing_or_out_of_range_draw_indices_rejected():
    with pytest.raises(ValueError):
        analysis.paired_interval([1, 0], [0, 0], np.array([[0, 2]]))
    with pytest.raises(ValueError):
        analysis.paired_interval([1, 0], [0, 0], np.array([[0]]))


def test_no_discordance_all_correct_has_no_benefit_or_equivalence_claim():
    rows, records = all_same(8, 1)
    result = analysis.summarize(rows, records, samples=100)
    p = result["primary_result"]
    assert p["difference"] == 0 and p["ci95"] == [0., 0.]
    assert p["exact_mcnemar"]["p_two_sided"] == 1.
    gate = result["quantitative_advancement"]
    assert gate["quantitative_rule_met"] is False
    assert gate["controller_ready"] is False
    transition = result["contrasts"][analysis.PRIMARY]["primary_em_transitions"]
    assert transition["negative_wrong"] == 0 and transition["fix_rate_among_negative_wrong"] is None
    assert transition["correct_to_correct"] == 8


def test_all_wrong_has_undefined_harm_denominator():
    rows, records = all_same(8, 0)
    result = analysis.summarize(rows, records, samples=100)
    t = result["contrasts"][analysis.PRIMARY]["primary_em_transitions"]
    assert t["negative_correct"] == 0
    assert t["harm_rate_among_negative_correct"] is None
    assert t["wrong_to_wrong"] == 8


def test_transitions_and_paired_point_estimate_use_sources_not_four_presentations():
    rows, records = fixture()
    result = analysis.summarize(rows, records, samples=400)
    t = result["contrasts"][analysis.PRIMARY]["primary_em_transitions"]
    assert result["n_source_clusters"] == 4 and result["n_records"] == 16
    assert (t["fixes"], t["harms"], t["correct_to_correct"], t["wrong_to_wrong"]) == (1, 1, 1, 1)
    assert t["fix_rate_among_negative_wrong"] == t["harm_rate_among_negative_correct"] == .5
    assert result["primary_result"]["difference"] == 0
    assert result["primary_result"]["exact_mcnemar"]["discordant_pairs"] == 2


def test_significant_adverse_effect_does_not_pass_benefit_gate():
    rows, records = fixture({a: [int(a == "degraded_256")] * 20 for a in analysis.ACTIONS})
    result = analysis.summarize(rows, records, samples=100)
    assert result["primary_result"]["exact_mcnemar"]["p_two_sided"] < .05
    assert result["primary_result"]["difference"] == -1
    assert result["quantitative_advancement"]["quantitative_rule_met"] is False


def test_quantitative_pass_is_never_semantic_pass_or_controller_ready():
    rows, records = fixture({a: [int(a == "native_256")] * 20 for a in analysis.ACTIONS})
    result = analysis.summarize(rows, records, samples=100)
    gate = result["quantitative_advancement"]
    assert gate["quantitative_rule_met"] is True
    assert gate["semantic_screen_status"] == "pending_separate_condition_blinded_artifact"
    assert gate["overall_advancement_status"] == "not_established_pending_semantic_screen"
    assert gate["automatic_controller_training"] is gate["controller_ready"] is False


def test_every_quantitative_boundary_is_independently_required():
    base = {"difference": .02, "ci95": [.001, .04]}
    passing = analysis.advancement_checks(base, {"p_two_sided": .05}, .8, .8)
    assert passing["quantitative_rule_met"]
    cases = [
        ({"difference": .019999, "ci95": [.001, .04]}, .05, .8, .8),
        (base, .0500001, .8, .8),
        ({"difference": .02, "ci95": [0., .04]}, .05, .8, .8),
        (base, .05, .799, .8),
    ]
    for primary, p, native, direct in cases:
        assert not analysis.advancement_checks(primary, {"p_two_sided": p}, native, direct)["quantitative_rule_met"]


def test_cost_adds_direct_only_for_regional_decision_state_and_memory_is_max():
    rows, records = fixture()
    group = analysis.validate_table(rows, records)[0]
    assert analysis.cost(group, "native_256") == 2
    assert analysis.cost(group, "native_256", mode="decision_state") == 9
    assert analysis.cost(group, "degraded_256", mode="decision_state") == 10
    assert analysis.cost(group, "native_256", "peak_memory_gib", "decision_state") == 12
    assert analysis.cost(group, "native_256", "peak_reserved_gib", "decision_state") == 16
    assert analysis.cost(group, "highres", mode="decision_state") == 11
    assert analysis.cost(group, "direct_256", mode="decision_state") == 7
    assert analysis.cost(group, "native_256", "generated_tokens", "decision_state") == 6
    assert analysis.cost(group, "native_256", "input_tokens", "decision_state") == 400
    with pytest.raises(ValueError):
        analysis.cost(group, "native_256", mode="oracle")


def test_latency_observations_p95_and_total_are_preserved():
    rows, records = fixture()
    result = analysis.summarize(rows, records, samples=100)
    a = result["actions"]["native_256"]
    assert a["observations"]["elapsed_s"] == [2., 3., 4., 5.]
    assert a["costs"]["standalone"]["elapsed_s"] == {
        "n": 4, "mean": 3.5, "median": 3.5, "min": 2., "p95": 4.85, "max": 5.}
    assert a["costs"]["decision_state"]["elapsed_s"]["mean"] == 12.
    assert a["observations"]["processor_observer_elapsed_s"] == [.1] * 4
    assert a["costs"]["standalone"]["processor_observer_elapsed_s"]["mean"] == .1
    assert a["costs"]["decision_state"]["processor_observer_elapsed_s"]["mean"] == .2
    assert result["standalone_measured_action_time_sum_s"] == sum(r["elapsed_s"] for r in records.values())


def invalidate(r, raw):
    r.update(response=raw, predicted_answer=None, normalized_prediction=None,
             parse_valid=False, primary_em=0., official_anls=0.)


def test_semantic_review_union_keeps_equal_grade_and_both_invalid_raw_changes():
    rows, records = fixture()
    records["example-2", "native_256"].update(predicted_answer="GOLD", response="ANSWER: GOLD")
    # index2: both correct but different raw parsed strings must be audited.
    invalidate(records["example-3", "native_256"], "ANSWER:")
    invalidate(records["example-3", "degraded_256"], "ANSWER: ")
    result = analysis.summarize(rows, records, samples=100)
    a = result["native_degraded_response_agreement"]
    assert a["semantic_audit_required_ids"] == [f"example-{i}" for i in range(4)]
    assert a["raw_changed_with_invalid_parse_ids"] == ["example-3"]
    assert a["audit_cases_by_automatic_primary_score"]["both_correct"] == ["example-2"]
    assert a["audit_cases_by_automatic_primary_score"]["both_wrong"] == ["example-3"]
    assert a["semantic_judgments_inferred"] is False


def test_valid_truncated_output_stays_scored_and_flagged():
    rows, records = all_same(3, 1)
    records["example-0", "native_256"].update(generation_truncated=True, generated_tokens=64)
    result = analysis.summarize(rows, records, samples=100)
    a = result["actions"]["native_256"]
    assert a["metrics"]["primary_em"] == 1
    assert a["truncated_generations"] == a["valid_truncated_answers"] == 1


@pytest.mark.parametrize("mutation,match", [
    (lambda r, d: d.pop(("example-1", "highres")), "Complete four-action"),
    (lambda r, d: r[1].update(source_cluster_id=r[0]["source_cluster_id"]), "Duplicate"),
    (lambda r, d: d["example-0", "native_256"].update(action="highres"), "identity"),
    (lambda r, d: d["example-0", "native_256"].update(status="error"), "Failed"),
    (lambda r, d: d["example-0", "native_256"].update(primary_em=.5), "binary"),
    (lambda r, d: d["example-0", "native_256"].update(official_anls=float("nan")), "bounded"),
    (lambda r, d: d["example-0", "native_256"].pop("generation_truncated"), "format flags"),
    (lambda r, d: d["example-0", "native_256"].update(generation_truncated=True), "Truncation"),
    (lambda r, d: d["example-0", "native_256"].update(elapsed_s=0), "latency"),
    (lambda r, d: d["example-0", "native_256"].update(processor_observer_elapsed_s=3), "Observer"),
    (lambda r, d: d["example-0", "native_256"].update(peak_reserved_gib=1), "memory"),
    (lambda r, d: d["example-0", "native_256"].update(input_tokens=128), "token"),
    (lambda r, d: d["example-0", "native_256"].update(generated_tokens=True), "token"),
])
def test_statistic_layer_rejects_corrupted_table(mutation, match):
    rows, records = fixture()
    mutation(rows, records)
    with pytest.raises(ValueError, match=match):
        analysis.summarize(rows, records, samples=10)


def test_locked_inference_settings_cannot_be_replaced_by_anls_or_small_bootstrap():
    c = config()
    analysis.validate_analysis_plan(c)
    for key, val in [("primary_metric", "official_anls"), ("bootstrap_samples", 9999),
                     ("bootstrap_seed", 20260926), ("alpha", .1)]:
        changed = copy.deepcopy(c); changed["analysis_plan"][key] = val
        with pytest.raises(ValueError, match="statistical plan"):
            analysis.validate_analysis_plan(changed)


def test_engineering_summary_withholds_metrics_tests_gates_and_never_bootstraps(monkeypatch):
    rows, records = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("Engineering must not bootstrap")
    monkeypatch.setattr(analysis, "bootstrap_draws", forbidden)
    result = analysis.summarize(rows, records, engineering=True)
    assert result["status"] == "completed_engineering_technical_check"
    assert "primary_result" not in result and "contrasts" not in result
    assert "quantitative_advancement" not in result
    assert all("metrics" not in a for a in result["actions"].values())
    assert result["quality_scores_reported"] is False


def validated_stub(tmp_path, monkeypatch, rows, records, role):
    manifest, lock = tmp_path/"manifest.jsonl", tmp_path/"lock.json"
    run_dir = tmp_path/"run";run_dir.mkdir()
    manifest.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    lock.write_text("{}", encoding="utf-8")
    run = {"role": role, "config": config(), "fingerprint": "synthetic",
           "model": {"synthetic": True}, "runtime": {"synthetic": True}}
    completed = {"records": len(records), "elapsed_s": 123.,
                 "elapsed_s_scope": "Synthetic segment only", "records_sha256": "synthetic",
                 "sum_measured_action_elapsed_s": sum(x["elapsed_s"] for x in records.values()),
                 "finished_utc": "2026-09-27T00:00:00Z"}
    for name, obj in [("run.json", run), ("completed.json", completed)]:
        (run_dir/name).write_text(json.dumps(obj), encoding="utf-8")
    (run_dir/"records.jsonl").write_text("\n".join(map(json.dumps, records.values())), encoding="utf-8")
    # These local provenance stubs are never imported/executed.
    (tmp_path/"experiments").mkdir()
    for name in ("dude_replication.py", "dude_metrics.py"):
        (tmp_path/"experiments"/name).write_text("# synthetic provenance only\n", encoding="utf-8")
    monkeypatch.setattr(analysis, "PROJECT", tmp_path)
    calls = []
    def validator(m, l, r, role="main"):
        calls.append((m, l, r, role))
        return rows, records, run, completed
    monkeypatch.setattr(analysis, "validate_completed_run", validator)
    return manifest, lock, run_dir, calls


def test_real_analysis_entry_requires_main660_after_validator(tmp_path, monkeypatch):
    rows, records = fixture()
    manifest, lock, run_dir, calls = validated_stub(tmp_path, monkeypatch, rows, records, "main")
    with pytest.raises(ValueError, match="exactly 660"):
        analysis.analyze(manifest, lock, run_dir)
    assert calls[0][3] == "main"


def test_completion_validator_failure_propagates_before_any_statistics(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("Incomplete locked run")
    def forbidden(*args, **kwargs):
        raise AssertionError("No summary before validation")
    monkeypatch.setattr(analysis, "validate_completed_run", reject)
    monkeypatch.setattr(analysis, "summarize", forbidden)
    with pytest.raises(ValueError, match="Incomplete locked"):
        analysis.analyze(tmp_path/"manifest", tmp_path/"lock", tmp_path/"run")


def test_engineering_requires_explicit_role_and_writes_only_technical_artifacts(tmp_path, monkeypatch):
    rows, records = fixture()
    manifest, lock, run_dir, calls = validated_stub(tmp_path, monkeypatch, rows, records, "engineering")
    result = analysis.analyze(manifest, lock, run_dir, engineering=True)
    out = tmp_path/"analysis"
    analysis.write_artifacts(result, out)
    assert calls[0][3] == "engineering"
    assert set(p.name for p in out.iterdir()) == {"summary.json", "report.md"}
    assert "primary_result" not in json.loads((out/"summary.json").read_text())
    assert "primary tests" in (out/"report.md").read_text()
    with pytest.raises(ValueError, match="absent or empty"):
        analysis.write_artifacts(result, out)


def test_main660_outputs_retain_completion_errors_hashes_and_figures(tmp_path, monkeypatch):
    rows, records = all_same(660, 1)
    manifest, lock, run_dir, calls = validated_stub(tmp_path, monkeypatch, rows, records, "main")
    (run_dir/"errors.jsonl").write_text('{"phase":"synthetic historical runtime event"}\n', encoding="utf-8")
    result = analysis.analyze(manifest, lock, run_dir)
    assert result["status"] == "completed_primary_replication"
    assert result["bootstrap"]["samples"] == 10000
    assert result["bootstrap"]["seed"] == 20260927
    assert result["n_records"] == 2640
    assert result["completion"]["elapsed_s_scope"] == "Synthetic segment only"
    assert result["runtime_error_log_entries"] == 1
    assert result["provenance"]["records_sha256"] == hashlib.sha256((run_dir/"records.jsonl").read_bytes()).hexdigest()
    out = tmp_path/"main-analysis"
    analysis.write_artifacts(result, out)
    actual = json.loads((out/"summary.json").read_text())
    assert len(actual["figures"]["files"]) == 6
    for relative, expected in actual["figures"]["files"].items():
        assert hashlib.sha256((out/relative).read_bytes()).hexdigest() == expected
    assert len(actual["actions"]["native_256"]["observations"]["elapsed_s"]) == 660
    assert "not establish equivalence" in (out/"report.md").read_text()
