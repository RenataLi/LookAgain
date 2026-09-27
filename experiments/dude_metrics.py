"""Prospective DUDE extractive scoring; custom primary EM, official scalar ANLS.

Only the non-list scalar path is supported, not a full DUDE benchmark evaluator.
The response parser is the unchanged LookAgain parser. Source validation of
gold/variants is a separate pre-inference audit; this module enforces provenance
membership, not semantic correctness. No TAT-QA normalization is used.

levenshtein_distance and the ANLS arithmetic reproduce the author's MIT code:
https://github.com/Jordy-VL/DUDEeval/blob/8bd6deac0f9747e344338e3850f02bf76744d888/evaluate_submission.py

MIT License
Copyright (c) 2023 Jordy Van Landeghem
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lookagain.actions import extract_answer

UPSTREAM_COMMIT = "8bd6deac0f9747e344338e3850f02bf76744d888"
UPSTREAM_SHA256 = {
    "evaluate_submission.py": "df11012257fc9650af0d696be8b171d27ed9eebdc659dd9dcb660233d8806f4d",
    "LICENSE": "d9f984c520481c141d8590a06457c61cede787270e892e8d2b7ce8fa7d51b066",
}


def _strings(values, name, allow_empty=False):
    if (not isinstance(values, (list, tuple)) or (not allow_empty and not values)
            or any(not isinstance(x, str) or not x.strip() for x in values)):
        raise ValueError(f"{name} must contain nonempty original strings")
    return tuple(values)


def normalize_primary(text):
    """Casefold + collapsed whitespace only; preserve punctuation, units, signs."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(text.casefold().split())


def score_primary(prediction, validated_answers):
    answers = _strings(validated_answers, "validated_answers")
    normalized = normalize_primary(prediction)
    return float(bool(normalized) and any(normalized == normalize_primary(x) for x in answers))


def levenshtein_distance(s1, s2):
    # Unchanged upstream scalar helper; no third-party dependency.
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def score_official_anls(prediction, official_answers):
    """Author's get_NLS(..., threshold=.5) for a scalar prediction.

Distance uses lower/strip/whitespace collapse; its denominator instead uses
RAW upper-case lengths. Similarity exactly .5 is retained. Only original
`answers` are passed: upstream ignores `answers_variants`. This function does
not add parser filtering or correct upstream whitespace/Unicode quirks.
"""
    if not isinstance(prediction, str):
        raise TypeError("prediction must be one string")
    answers = _strings(official_answers, "official_answers")
    values = []
    for answer in answers:
        gold = " ".join(answer.strip().lower().split())
        predicted = " ".join(prediction.strip().lower().split())
        distance = levenshtein_distance(gold, predicted)
        length = max(len(answer.upper()), len(prediction.upper()))
        values.append(0.0 if length == 0 else float(distance) / float(length))
    result = 1 - min(values)
    return 0.0 if result < 0.5 else result


def validate_reference_sets(official_answers, validated_answers, official_answer_variants=()):
    """Require canonical support and literal membership in the source annotation."""
    original = _strings(official_answers, "official_answers")
    variants = _strings(official_answer_variants, "official_answer_variants", allow_empty=True)
    validated = _strings(validated_answers, "validated_answers")
    if len(original) != 1:
        raise ValueError("This cohort requires exactly one original extractive answer")
    if original[0] not in validated:
        raise ValueError("Canonical answer must be source-validated; do not repair gold by replacing it")
    if not set(validated) <= set(original + variants):
        raise ValueError("Validated answers must be unchanged strings from original answers/variants")
    return original, validated


def score_response(response, official_answers, validated_answers, official_answer_variants=()):
    """Evaluate a full response; invalid parsing is a failure in every metric.

The caller must bind source-only validation evidence for `validated_answers`.
Passing all metadata variants without that audit violates the protocol.
"""
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    original, validated = validate_reference_sets(official_answers, validated_answers, official_answer_variants)
    prediction = extract_answer(response)
    em = score_primary(prediction, validated) if prediction is not None else 0.0
    anls = score_official_anls(prediction, original) if prediction is not None else 0.0
    return {
        "predicted_answer": prediction, "parse_valid": prediction is not None,
        "answer_format": "marked" if re.search(r"(?im)^ANSWER:", response) else "bare",
        "normalized_prediction": normalize_primary(prediction) if prediction is not None else None,
        "primary_em": em, "strict_em": em, "correct": bool(em),
        "official_anls": anls, "anls": anls,
    }
