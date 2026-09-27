# Reproduce the numerical benchmark

From the repository root with Python 3.11 or newer:

```sh
python -S scripts/verify_release.py
python -S scripts/reproduce_results.py
```

Both commands use the standard library. The first verifies the distributed files against
their SHA-256 manifest. The second performs **4,806 scalar comparisons across 124 aligned
observations and 22 strategies**, without downloading model weights or source documents.
Each replay writes a fresh directory under `generated/`; `--output <new-directory>` selects
another destination.

## What the replay calculates

- Custom exact-match and scalar ANLS means for all 22 instruction/policy combinations.
- Mean, median, p95, minimum and maximum of eight cost measurements.
- Invalid and truncated answer counts, and uniform-crop expectations.
- Paired policy differences, gains and harms, and results for each of five folds.
- The instruction/ranking interaction and a seeded 10,000-sample fixed-policy prompt bootstrap.

The primary comparison is **+1.612903 percentage points** for full ranking versus prompted
selection, with eight gains and six harms. Fold assignments preserve the original experiment.
All 124 observations are reused development data; comparisons describe this study population.

## Numerical inputs

[`scores.json`](../reports/benchmark/scores.json) contains aligned numeric observations.
[`expected_summary.json`](../reports/benchmark/expected_summary.json) contains the corresponding
expected statistics. Values are numbers, booleans or nulls; array position pairs an observation
across strategies. Questions, answer strings, source identifiers and source-document content
remain outside this distribution.

This workflow recomputes arithmetic from exported scores and assembled measured costs.
Raw-answer scoring, feature extraction, ranker fitting and model generation are separate
stages implemented in the experiment code. The [inference guide](running_inference.md)
provides the environment, a pinned-model smoke run and the study module map.
The [model description](refinement_model.md) specifies the learned representation and objective.

## Tests and figures

```sh
python -m pip install "pytest>=8,<10"
python -m pytest -q tests/test_numeric_replay.py
```

Focused tests exercise changed scores, costs, folds and expected values; missing observations;
nonfinite numbers; and unexpected fields. The GitHub Actions workflow runs the complete CPU
suite on Python 3.11 and 3.12 and repeats both release checks.

To regenerate the result chart with Matplotlib installed:

```sh
python scripts/plot_public_results.py
```
