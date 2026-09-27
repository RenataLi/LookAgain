"""DUDE scalar scoring semantics, provenance membership and frozen parsing."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "dude_metrics", Path(__file__).resolve().parents[1] / "experiments/dude_metrics.py")
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def test_primary_casefold_whitespace_but_preserves_signs_units_and_punctuation():
    assert metrics.score_primary(" STRASSE\t  5\nkg ", ["Straße 5 kg"]) == 1
    for wrong in ("5", "-5 kg", "5 kg.", "the 5 kg", "$5 kg"):
        assert metrics.score_primary(wrong, ["5 kg"]) == 0
    assert metrics.score_primary("", ["5 kg"]) == 0


def test_official_similarity_boundary_is_inclusive():
    assert metrics.score_official_anls("ab", ["ac"]) == 0.5
    assert metrics.score_official_anls("ab", ["cd"]) == 0
    assert metrics.score_official_anls("abc", ["adc"]) == pytest.approx(2 / 3)


def test_official_denominator_uses_raw_uppercase_lengths_not_normalized_lengths():
    # Normalized edit distance is 1, but original gold length is 6, not 3.
    assert metrics.score_official_anls("a c", [" a  b "]) == pytest.approx(5 / 6)
    # Unicode upper-case expansion also affects the author's denominator.
    assert metrics.score_official_anls("a", ["ß"]) == 0.5
    assert metrics.score_official_anls("STRASSE", ["Straße"]) == pytest.approx(5 / 7)


def test_official_scalar_retains_author_empty_prediction_quirk():
    assert metrics.score_official_anls("", ["a"]) == 0
    assert metrics.score_official_anls("", ["   a   "]) == pytest.approx(6 / 7)
    # Our response adapter explicitly rejects invalid/empty parser output.
    score = metrics.score_response("", ["   a   "], ["   a   "])
    assert score["parse_valid"] is False
    assert score["official_anls"] == score["primary_em"] == 0


def test_original_variants_only_affect_custom_primary_when_source_approved():
    score = metrics.score_response("ANSWER: different", ["canonical"],
                                   ["canonical", "different"], ["different"])
    assert score["primary_em"] == 1
    assert score["official_anls"] == metrics.score_official_anls("different", ["canonical"])
    assert score["official_anls"] == 0
    unapproved = metrics.score_response("different", ["canonical"], ["canonical"], ["different"])
    assert unapproved["primary_em"] == 0


@pytest.mark.parametrize("original,validated,variants", [
    (["a"], ["b"], ["b"]), (["a"], ["a", "invented"], []),
    (["a", "b"], ["a", "b"], []), (["a"], ["a"], [""]),
    ([], ["a"], []), (["a"], [], []), (["a"], ["A"], []),
])
def test_bad_or_silently_repaired_gold_is_rejected(original, validated, variants):
    with pytest.raises(ValueError):
        metrics.score_response("a", original, validated, variants)


@pytest.mark.parametrize("response", ["ANSWER:", "ANSWER: a\nANSWER: a", "thinking\nmore", "<a>"])
def test_invalid_full_response_is_not_searched_for_a_gold_substring(response):
    score = metrics.score_response(response, ["a"], ["a"])
    assert score["parse_valid"] is False
    assert score["primary_em"] == score["official_anls"] == 0


def test_unique_final_marker_and_bare_sentence_use_unchanged_parser():
    assert metrics.score_response("I considered the source.\nANSWER: a", ["a"], ["a"])["primary_em"] == 1
    assert metrics.score_response("The answer is a.", ["a"], ["a"])["primary_em"] == 0


def test_edit_distance_and_multiple_official_references_scalar_utility():
    assert metrics.levenshtein_distance("kitten", "sitting") == 3
    assert metrics.levenshtein_distance("", "text") == 4
    assert metrics.score_official_anls("target", ["unrelated", "target"]) == 1
