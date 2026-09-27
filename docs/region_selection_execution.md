# Given-page region-selection workflow

This guide describes the completed prompted-selection pilot and its implementation.
The [study index](../reports/README.md) summarizes its results; the
[protocol](region_selection_protocol.md) specifies the comparison. The current
[CPU replay](reproduce.md) covers the later 124-source numerical release.

## Source separation and requests

The original pilot reserved 2 engineering, 64 development and 60 evaluation
source clusters. The public [membership projection](../reports/cohort_splits.json)
retains integer-indexed split sizes and exclusion relationships. It does not
distribute source identifiers, questions or references, or independently repeat
the original semantic source audit.

Each provided answer page has nine overlapping windows spanning half its width
and half its height. The frozen VLM selects one ID from the overview and question.
The answer request receives the overview and selected crop. Reference answers,
OCR transcripts and annotated evidence boxes are excluded from selection.

The original chronology was:

1. Reserve the three source cohorts and separate observations from labels.
2. Check direct, selector, nine native crops, nine degraded crops and highres
   for every source: 2,646 actual CPU processor requests.
3. Lock and complete 26 engineering calls; then lock and collect 832 development calls.
4. Freeze development-only confidence quantiles without using answer labels.
5. Lock evaluation and collect all 780 calls before inspecting its quality.
6. Validate the complete requests, statistics and execution identities.

The model was Qwen3-VL-4B-Instruct revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, using greedy BF16/SDPA inference.
Six synthetic warmups per role were separate from experimental calls.

## Exercise the implementation

```sh
python -m pytest tests/test_region_selection_core.py tests/test_region_selection_execution.py tests/test_region_selection_analysis.py tests/test_region_result_check.py -q
```

The tests cover request construction, strict parsing, geometry, lock integrity,
calibration and paired policy arithmetic with synthetic inputs. The membership
test checks the exported projection and labels its scope explicitly.

[Preparation](../experiments/prepare_region_selection.py),
[execution](../experiments/run_region_selection.py),
[analysis](../experiments/analyze_region_selection.py) and the
[raw-record checker](../experiments/check_region_results.py) remain available.
Their full historical workflows require local source media, manifests, original
locks, processor captures and raw answers. These inputs, including the original
documentation bytes covered by historical bindings, are not part of this numerical
release. Use their `--help` interfaces alongside the [data preparation guide](dude_data_preparation.md)
when designing a separately validated experiment. Existing byte checks should
not be weakened to make substitute inputs appear identical to the original run.

## Interpret selection and cost

Uniform quality averages nine outcomes within each source; it represents one
uniformly chosen crop, not nine online calls. The paired sample size is 60 rather
than 540. The finite-bank oracle uses answer labels and is a diagnostic upper bound.

Selected-policy cost includes the selector and chosen answer. Confidence gates
also pay for a direct answer before deciding whether to zoom. These are assembled
measured components, not separately timed end-to-end policy invocations. Sequential
memory is a maximum, not a sum; allocator reservation is not physical residency.

The 60 evaluation sources have now been inspected and reused in later development.
They no longer provide untouched confirmation for subsequent improvements.
