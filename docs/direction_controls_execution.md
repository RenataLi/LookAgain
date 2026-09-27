# Direction controls: engineering and execution notes

This guide records the original local experiment workflow. Its commands require the historical source manifests, raw records and identity artifacts, which are not bundled in this numerical release. See the [study index](../reports/README.md) and [current CPU replay](reproduce.md) for the published materials.

This file records implementation checks and interpretation constraints. Main results belong in `reports/direction400/`; a completed outcome report requires all 400 images and 6,800 generation records. The two-image engineering smoke is never pooled with the primary panel.

## Completed primary run

The uninterrupted run completed all **400 images and 6,800 actions in 52.5 minutes after warm-up**. Strict analysis verified complete unique coverage, source-image hashes, recorded pixel/prompt control identities and the run fingerprint. There were no execution errors, invalid answer formats or truncated generations. Export rechecked that the base inference package, experiment runner and configuration still matched the recorded run. Peak allocated GPU memory was 10.23965 GiB.

Fresh Direct and all four Named branches reproduced 2,000/2,000 normalized answers and correctness labels from v1. This repeat supports reproducibility on the same panel and runtime; it is not new held-out evidence. The complete source and analysis test suite passed 50 checks. The three planned contrasts and all subgroup/cost results are retained, regardless of direction.

## Before primary analysis

- The four prompts, action/quadruple schedule, lexical subgroup, three primary contrasts and bootstrap rules were locked in `configs/direction_controls.json` at 2026-09-25 17:17:18 UTC, before smoke or main outcomes. This is an internal protocol lock, not external preregistration.
- The v1 inference package was preserved byte-for-byte. Direct and named-crop construction in v2 were compared with the original v1 helper on a real source image: messages and resulting pixel bytes matched.
- A separate two-image, 34-call smoke completed. All calls parsed and stayed within the output cap. The three true-crop conditions used the same RGB hash for a given quadrant; all sham second images matched their direct overview; named and sham prompts matched.
- The initial complete CPU suite passed 42 checks, including eight new runner tests. They exercise independent branch construction, pixel identities, original-helper equivalence, execution orchestration, strict resumption, changed runtime/configuration rejection, and duplicate/error handling. Any later analysis tests are recorded with the completed report.
- Eight additional analysis tests passed, bringing the complete suite to **50 passing tests**. They check an analytically known clustered interval, contrast signs, zero paired variance for equal conditions, transition denominators, invalid direction outputs, height-matched sham pairing, direct-plus-follow-up timing/max memory and missing/pixel-mismatched records. Separate checks of the analyzer, figures and gallery found no blocking calculation issue.
- Evidence aggregation followed completion of the main run. Interim monitoring was limited to progress, errors, hardware use and engineering integrity; prompts and inference settings remained fixed.

## What the cost measurements include

Calls are synchronized with the GPU. The measured path includes decoding the source image, resizing, hashing the constructed RGB views, token/image processing, device transfers, generation, decoding and diagnostic log-probability extraction. It excludes model loading, synthetic warm-up and JSONL writes. A sequential follow-up policy pays for the direct answer plus the chosen follow-up; its peak allocated memory is the larger of their two peaks.

The sham implementation first calls the original crop builder and then replaces its additional image with `overview.copy()`. Consequently, it performs an unused crop construction on the CPU; that overhead is included in the recorded sham latency. It is intentionally left unchanged during the experiment. These measurements describe this instrumented implementation and are not a deployment-optimized speed comparison.

The exact copied overview can have a different rounded shape from a true crop despite their nominally equal visual-token budgets. Realized grids and prompt lengths are recorded; equal nominal tokens do not establish equal computation. Warm-up covers three representative aspect ratios with direct and named-crop calls, not every realized image shape or prompt length. Generation uses greedy decoding, but hardware/software reproducibility must be measured rather than assumed.

Light local CPU work overlapped the GPU run: unit tests, source-image inspection, file checks and report preparation. No additional model inference was launched during the main run. Consequently, this is not an isolated hardware throughput benchmark; within-image branch shuffling reduces systematic ordering effects but does not remove all timing variation.

## Qualitative display

The offline gallery displays the source image, marked quadrant boxes, true-crop/sham previews, direct answer, all 16 follow-up slots, original correctness labels and total sequential cost. Its deterministic selection cycles through outcome strata; it is an illustrative inspection sample, not a representative estimate. The full raw records and quantitative analysis remain the primary evidence.

The gallery is static and contains no remote assets or executable JavaScript. Its structure, escaping, slot coverage and embedded assets were checked on the smoke run. The PDF and plots received separate rendered-image checks.

## Reproduction

From the repository root, with the pinned model and prepared data already available:

```text
python experiments/direction_controls.py --manifest data/gqa_dev400/manifest.jsonl --model-dir models/qwen3-vl-4b --config configs/direction_controls.json --output runs/direction400
python experiments/analyze_direction_controls.py --run runs/direction400 --manifest data/gqa_dev400/manifest.jsonl --output reports/direction400 --v1-run reports/pilot400 --v1-diagnostics reports/pilot400/diagnostics.json
python experiments/plot_direction_controls.py --summary reports/direction400/summary.json --output reports/direction400
python experiments/render_direction_gallery.py --run runs/direction400 --manifest data/gqa_dev400/manifest.jsonl --output ../lookagain-local-gallery/direction.html
```

Use a separate run directory with `--limit 2` for smoke validation. Analysis of that declared small run requires `--allow-smoke` and is labeled engineering-only. The default analyzer rejects an incomplete or undersized primary panel. It also validates source-image files: the matching manifest must reside beside the original prepared images, even when recomputing statistics from archived records. It does not load model weights. Run the strict analyzer before rendering the gallery from the same unchanged records. The image-containing gallery must be outside the code project, as shown above.

To rebuild the report, install optional report dependencies and provide the completed summary, figures and run-bound interpretation:

```text
python -m pip install -e ".[report]"
python experiments/build_direction_note.py --summary reports/direction400/summary.json --interpretation reports/direction400/interpretation.json --output local-gallery/direction_note.pdf
```

The prose is specific to this experiment. A new run requires a new interpretation and case audit, not merely substituted numeric values.
