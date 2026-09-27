"""Parity-check harness integrity; no network or upstream fixture dependency."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "dude_metric_parity_tests", Path(__file__).resolve().parents[1] / "experiments/check_dude_metric_parity.py")
parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parity)


def test_fixture_count_seed_and_boundary_cases_are_reproducible():
    cases = parity.synthetic_cases()
    assert len(cases) == 1005
    assert cases == parity.synthetic_cases()
    assert cases[:5] == [([" a  b "], "a c"), (["ac"], "ab"), (["Straße"], "STRASSE"),
                         (["   a   "], ""), (["a"], "a")]
    assert all(len(answers) == 1 and answers[0].strip() for answers, _ in cases)


@pytest.mark.parametrize("bad_evaluator", [False, True])
def test_pin_rejection_precedes_ast_parse_and_compile(monkeypatch, bad_evaluator):
    evaluator, license_text = b"valid fixture bytes", b"fixture license"
    monkeypatch.setattr(parity, "UPSTREAM_PINS", {"evaluate_submission.py": parity.digest(evaluator), "LICENSE": parity.digest(license_text)})
    monkeypatch.setattr(parity.ast, "parse", lambda *args, **kwargs: pytest.fail("Unpinned code reached AST parsing"))
    if bad_evaluator:
        evaluator += b"changed"
    else:
        license_text += b"changed"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        parity.compile_pinned_reference(evaluator, license_text)


def test_only_selected_functions_are_compiled_not_imports_or_top_level_code(monkeypatch):
    evaluator = b'''import nonexistent_module_that_must_not_be_imported
raise RuntimeError("top level must not execute")
def levenshtein_distance(a,b):
    return 2
def parse_answers(pred):
    logging.warning("silenced test log")
    return pred
def get_NLS(gold,pred,threshold):
    return threshold + levenshtein_distance(gold, parse_answers(pred))
def unrelated():
    raise RuntimeError("unused")
'''
    license_text = b"synthetic license"
    monkeypatch.setattr(parity, "UPSTREAM_PINS", {"evaluate_submission.py": parity.digest(evaluator), "LICENSE": parity.digest(license_text)})
    reference = parity.compile_pinned_reference(evaluator, license_text)
    assert reference(["a"], "b", 0.5) == 2.5
    assert "unrelated" not in reference.__globals__


def test_mismatching_metric_is_reported_not_tolerated_or_dropped():
    result = parity.compare_cases(lambda *_: 0.0, [(["gold"], "gold"), (["gold"], "unrelated")])
    assert result["status"] == "FAIL" and result["checks"] == 2
    assert result["failures"][0]["fixture_index"] == 0
    assert result["maximum_absolute_error"] == 1


def test_nonfinite_reference_is_failure_even_when_nan_comparisons_are_false():
    result = parity.compare_cases(lambda *_: float("nan"), [(["gold"], "gold")])
    assert result["status"] == "FAIL"
    assert result["failures"] == [{"fixture_index": 0, "reason": "nonfinite_result"}]
