"""Exercise analysis CLI coverage and reporting without models or downloads."""

import json
from pathlib import Path

import pytest

from lookagain.cli import main


def _record(example_id, action, *, correct=True, parse_valid=True, truncated=False):
    return {
        "example_id": example_id,
        "image_id": f"image:{example_id}",
        "action": action,
        "status": "ok",
        "correct": correct,
        "elapsed_s": 1.0 if action == "direct" else 0.5,
        "peak_memory_gib": 2.0,
        "parse_valid": parse_valid,
        "generation_truncated": truncated,
    }


def _write_run(root: Path, planned, records, *, actions=("direct", "think")):
    run = root / "run"
    run.mkdir()
    metadata = {
        "config": {"actions": list(actions), "seed": 20260925},
        "example_ids": list(planned),
        "fingerprint": "a" * 64,
    }
    (run / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return run, root / "analysis"


def _analyze(monkeypatch, run, output):
    monkeypatch.setattr("sys.argv", ["lookagain", "analyze", "--run", str(run), "--output", str(output)])
    main()
    return (
        json.loads((output / "summary.json").read_text(encoding="utf-8")),
        (output / "report.md").read_text(encoding="utf-8"),
    )


def test_wholly_absent_example_marks_run_incomplete_and_warns_in_report(tmp_path, monkeypatch):
    run, output = _write_run(
        tmp_path, ["seen", "absent"],
        [_record("seen", "direct"), _record("seen", "think", correct=False)],
    )
    summary, report = _analyze(monkeypatch, run, output)

    assert summary["status"] == "incomplete_coverage"
    assert summary["coverage"]["complete_examples"] == 1
    assert summary["coverage"]["planned_examples"] == 2
    assert summary["coverage"]["wholly_absent_examples"] == ["absent"]
    assert summary["coverage"]["planned_completion_fraction"] == 0.5
    assert all(row["n_examples"] == 1 for row in summary["actions"].values())
    # A complete surviving pair must not make the full run look complete.
    assert "Status: **incomplete_coverage**" in report
    assert "Status: **complete**" not in report
    assert "1 planned examples have no recorded actions; this run is incomplete." in report
    assert "Planned examples: 2. Entirely absent: 1." in report
    assert "not a held-out benchmark result" in report


def test_partial_and_wholly_absent_examples_have_distinct_coverage(tmp_path, monkeypatch):
    run, output = _write_run(
        tmp_path, ["complete", "partial", "absent"],
        [_record("complete", "direct"), _record("complete", "think"),
         _record("partial", "direct")],
    )
    summary, report = _analyze(monkeypatch, run, output)

    assert summary["coverage"]["complete_examples"] == 1
    assert summary["coverage"]["excluded_examples"] == 1
    assert summary["coverage"]["excluded_groups"][0]["example_id"] == "partial"
    assert summary["coverage"]["excluded_groups"][0]["problems"] == {"think": "missing"}
    assert summary["coverage"]["wholly_absent_examples"] == ["absent"]
    assert summary["coverage"]["planned_completion_fraction"] == pytest.approx(1 / 3)
    assert "Planned examples: 3. Entirely absent: 1." in report


def test_unexpected_example_is_rejected_before_writing_outputs(tmp_path, monkeypatch):
    run, output = _write_run(
        tmp_path, ["planned"],
        [_record("foreign", "direct"), _record("foreign", "think")],
    )
    with pytest.raises(ValueError, match="outside the run manifest"):
        _analyze(monkeypatch, run, output)
    assert not output.exists()


def test_empty_records_report_absence_without_fabricated_accuracy(tmp_path, monkeypatch):
    run, output = _write_run(tmp_path, ["first", "second"], [])
    summary, report = _analyze(monkeypatch, run, output)

    assert summary["status"] == "incomplete_coverage"
    assert summary["coverage"]["wholly_absent_examples"] == ["first", "second"]
    assert summary["coverage"]["planned_completion_fraction"] == 0
    assert summary["actions"] == {}
    assert summary["baselines"] == {}
    assert "No complete paired groups" in report
    assert "| Method |" not in report
    assert "Entirely absent: 2" in report


def test_parse_and_truncation_diagnostics_and_prespecified_analysis_are_saved(tmp_path, monkeypatch):
    run, output = _write_run(
        tmp_path, ["first", "second"],
        [_record("first", "direct"),
         _record("first", "think", correct=False, parse_valid=False, truncated=True),
         _record("second", "direct", correct=False),
         _record("second", "think")],
    )
    summary, report = _analyze(monkeypatch, run, output)

    assert summary["status"] == "complete"
    assert summary["bootstrap"]["samples"] == 10000
    assert summary["bootstrap"]["seed"] == 20260925
    assert summary["expected_actions"] == ["direct", "think"]
    assert summary["run_fingerprint"] == "a" * 64
    assert summary["format_diagnostics"]["direct"] == {
        "records": 2, "invalid_answers": 0, "truncated_generations": 0,
    }
    assert summary["format_diagnostics"]["think"] == {
        "records": 2, "invalid_answers": 1, "truncated_generations": 1,
    }
    assert "| think | 2 | 1 | 1 |" in report
    assert "Upsampling introduces no new source detail" in report
    assert summary["actions"]["think"]["total_latency_s"]["mean"] == 1.5


def test_invalid_run_metadata_json_is_rejected_without_a_report(tmp_path, monkeypatch):
    run, output = _write_run(tmp_path, ["first"], [_record("first", "direct")])
    (run / "run.json").write_text('{"config":', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        _analyze(monkeypatch, run, output)
    assert not output.exists()


def test_missing_run_identity_does_not_emit_unattributed_results(tmp_path, monkeypatch):
    run, output = _write_run(
        tmp_path, ["first"], [_record("first", "direct"), _record("first", "think")]
    )
    metadata_path = run / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["fingerprint"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(KeyError, match="fingerprint"):
        _analyze(monkeypatch, run, output)
    assert not output.exists()
