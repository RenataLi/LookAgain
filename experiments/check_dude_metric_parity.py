"""Reproduce 1,005 scalar-ANLS comparisons against pinned local DUDEeval code.

No network, model, dataset, or outcome reads. Requires the author's unchanged
evaluate_submission.py and LICENSE from the commit recorded in dude_metrics.py.
Only three selected function definitions are compiled after BOTH hashes pass.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dude_metrics

SEED = 20260927
FUNCTIONS = {"levenshtein_distance", "parse_answers", "get_NLS"}
UPSTREAM_PINS = dict(dude_metrics.UPSTREAM_SHA256)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def compile_pinned_reference(evaluator_bytes, license_bytes):
    """No parsing/compilation occurs before both exact upstream pins match."""
    for name, blob in (("evaluate_submission.py", evaluator_bytes), ("LICENSE", license_bytes)):
        if digest(blob) != UPSTREAM_PINS[name]:
            raise ValueError(f"Pinned upstream {name} SHA256 mismatch")
    tree = ast.parse(evaluator_bytes.decode("utf-8"), filename="pinned_DUDEeval/evaluate_submission.py")
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if len(selected) != 3 or {node.name for node in selected} != FUNCTIONS:
        raise ValueError("Expected exactly the three scalar upstream function definitions")
    module = ast.Module(body=selected, type_ignores=[])
    # Upstream parse_answers logs empty predictions; logging is not a score input.
    namespace = {"logging": SimpleNamespace(warning=lambda *args, **kwargs: None)}
    exec(compile(module, "pinned_DUDEeval_scalar_functions", "exec"), namespace)
    return namespace["get_NLS"]


def synthetic_cases():
    """Same five explicit fixtures and 1,000 seeded strings as the initial check."""
    cases = [([" a  b "], "a c"), (["ac"], "ab"), (["Straße"], "STRASSE"),
             (["   a   "], ""), (["a"], "a")]
    generator = random.Random(SEED)
    alphabet = "abCD  \t\n.,-$%012ßİé"
    for _ in range(1000):
        gold = "".join(generator.choice(alphabet) for _ in range(generator.randrange(1, 30)))
        if not gold.strip():
            gold += "a"
        prediction = "".join(generator.choice(alphabet) for _ in range(generator.randrange(0, 35)))
        cases.append(([gold], prediction))
    return cases


def compare_cases(reference, cases, tolerance=1e-14):
    failures, maximum_error = [], 0.0
    for index, (answers, prediction) in enumerate(cases):
        expected = reference(answers, prediction, 0.5)
        actual = dude_metrics.score_official_anls(prediction, answers)
        error = abs(expected - actual)
        if not math.isfinite(expected) or not math.isfinite(actual):
            failures.append({"fixture_index": index, "reason": "nonfinite_result"})
            continue
        maximum_error = max(maximum_error, error)
        if error > tolerance:
            failures.append({"fixture_index": index, "expected": expected, "actual": actual, "absolute_error": error})
    return {"checks": len(cases), "failures": failures, "maximum_absolute_error": maximum_error,
            "absolute_tolerance": tolerance, "status": "PASS" if not failures else "FAIL"}


def verify(upstream_path, license_path):
    upstream = Path(upstream_path).read_bytes()
    license_text = Path(license_path).read_bytes()
    reference = compile_pinned_reference(upstream, license_text)
    cases = synthetic_cases()
    result = compare_cases(reference, cases)
    result.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Source-only scalar scorer verification; synthetic strings only; no model outcomes",
        upstream_commit=dude_metrics.UPSTREAM_COMMIT,
        upstream_evaluator_sha256=digest(upstream), upstream_license_sha256=digest(license_text),
        adapter_sha256=digest(Path(dude_metrics.__file__).read_bytes()),
        verification_script_sha256=digest(Path(__file__).read_bytes()),
        fixture_set_sha256=digest(json.dumps(cases, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
        seed=SEED, fixed_fixtures=5, seeded_synthetic_fixtures=1000,
        fixture_definition="Same five boundary/whitespace/Unicode cases and seeded alphabet generation as the initial work-directory 1005 check",
        compiled_functions=sorted(FUNCTIONS), upstream_imports_or_main_executed=False,
        method="Verify evaluator and license SHA256, then compile only the three selected top-level AST function definitions; compare their scalar get_NLS at threshold0.5 with the independent adapter",
        runtime={"python": sys.version.split()[0]},
        limitations=["Checks only scalar ANLS, not the complete DUDE evaluator, list matching or benchmark aggregation.",
                     "Does not validate source references, model parsing, custom primary EM or semantic cohort eligibility.",
                     "Preserves upstream Unicode/whitespace denominator behavior and inclusive similarity0.5; it does not endorse those choices."])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True, help="Local pinned evaluate_submission.py")
    parser.add_argument("--license", type=Path, required=True, help="Local pinned upstream LICENSE")
    parser.add_argument("--output", type=Path, required=True, help="JSON verification report")
    args = parser.parse_args()
    result = verify(args.upstream, args.license)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "checks", "maximum_absolute_error")}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
