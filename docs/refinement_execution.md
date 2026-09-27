# Reproduce the learned-ranking results

The public release supports CPU replay of the latest study's paired numerical
scores and measured costs. Start with the [research note](../reports/research_note.md),
[model specification](refinement_model.md) and [study protocol](refinement_protocol.md).

```sh
python -S scripts/reproduce_results.py
```

The [replay guide](reproduce.md) defines the exact checks and output files.
`reports/benchmark/scores.json` preserves unnamed source pairing, committed fold
membership, precomputed grades and costs; `expected_summary.json` supplies the
expected statistics. The replay recomputes aggregate arithmetic with the Python
standard library. Answer-text grading and ranker fitting belong to the full local
experiment pipeline described below.

## Experimental design

All **124 sources are reused development data**: the earlier 64 development and
60 evaluation sources. Five source folds keep a source's old labels out of its
own fitted ranker; the estimates describe this reused development cohort. The dataset supplies
the answer page. Region selection receives only that page and its question.

The reference execution used frozen Qwen3-VL-4B-Instruct revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, Transformers 4.57.6, PyTorch 2.8.0,
BF16/SDPA and greedy decoding on an RTX 5090. The new answer instruction was
evaluated on the same nine candidate windows used by the earlier study.

The original sequence separated preparation, CPU input checks, generation,
feature extraction and training:

1. Prepare the 124-source panel and two separate engineering sources; keep
   observation manifests distinct from evaluator labels.
2. Check all 2,646 possible requests with the actual CPU processor before generation.
3. Complete engineering generation and feature checks.
4. Extract main decoder features and fit five out-of-fold rankers using only
   archived old answer grades. Commit the chosen region IDs before new answers.
5. Collect all 1,612 main generations, then validate and analyze the complete bank.

Implementation entry points are [preparation](../experiments/prepare_refinement.py),
[execution](../experiments/refinement_execution.py),
[training](../experiments/train_refinement.py) and
[analysis](../experiments/analyze_refinement.py). Their command-line help documents
the required local inputs. The [inference guide](running_inference.md) provides
the environment and a self-contained GQA smoke-run entry point.

## Numerical replay and full execution

The published numerical inputs contain no document text, generated responses or
feature arrays. Full-data [coefficient sets](../artifacts/ranker_coefficients.json)
are available for inspecting the fitted models; their public performance estimate
comes from the out-of-fold experiment, not a separate full-data evaluation.

The historical strict path additionally needs the original page renders,
observation/label manifests, raw answer banks, raw and projected feature records,
decisions, processor captures and execution locks. These are local research
artifacts, not files supplied by this numerical release. Historical byte bindings
also cover the original documentation; this publication guide is adapted for the
distributed materials. A new experiment needs its own explicit source preparation,
validation and chronological commitments.

With complete local artifacts, the strict validators reconstruct requests and
check model/code/configuration identities, source hashes, action coverage,
projection, fold fits and predictions.

## Reading quality and compute

The main learned-policy contrast is descriptive: full-ranker/new versus
prompted-selector/new, with source-weighted means and separate results for each
of five folds. The fixed-policy old/new instruction comparison has a
paired development interval and uses the archived selector's region in both banks.
Invalid and truncated answers remain in their denominators.

Policy time assembles measured preprocessing, feature extraction, CPU ranking or
VLM selection, and answer generation. Feature time includes pooling and projection.
Offline fitting, loading and warmups have separate accounting. Reported policy
latency is assembled from these active-desktop measurements; sequential GPU peak is the maximum component
peak. The high-resolution full-page baseline remains part of every main comparison.
