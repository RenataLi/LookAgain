# Reproduce the evidence-availability diagnostic (v5)

This guide records the original local experiment workflow. Its commands require the historical source manifests, raw records and identity artifacts, which are not bundled in this numerical release. See the [study index](../reports/README.md) and [current CPU replay](reproduce.md) for the published materials.

This diagnostic varies overview resolution while holding an annotation-guided native region fixed. Its region is selected using reference-linked annotation. The reference text never enters the language prompt, but the location itself is privileged information. This is not a deployable crop selector or a held-out evaluation.

## Prepare and lock

Restore the original TAT-DQA training JSON/ZIP and the exact 200-DPI main100/smoke2 data described in [native-detail execution](native_detail_execution.md). Keep the original PDFs in the prepared source directory. Install the project and the optional `report` dependencies for PDF geometry parsing. Released thumbnail images are unsuitable for this diagnostic. Model weights, source PDFs and full page images are omitted from the source archive.

Run from the project root, substituting your local paths:

```text
python experiments/prepare_evidence_roi.py --raw data/tatdqa_raw --source-dir data/tatdqa_v3 --quality-mask reports/native100/source_quality_mask.json --output data/tatdqa_v5
python -m pytest tests -q
python experiments/audit_evidence_processor.py --manifest data/tatdqa_v5/smoke_manifest.jsonl --model-dir models/qwen3-vl-4b --output local-runs/evidence-processor.json
```

The preparation script first applies the prior source-quality mask, validates reference-linked OCR character/word mappings, maps OCR CropBox coordinates into the MediaBox-rendered image, and constructs one context region per report. It makes no model calls and reads no model outcomes. Unsupported geometry and invalid mappings are excluded with first-failure reasons. It copies accepted source PNGs byte for byte and records the full selection audit. The frozen cohort contains 85 main reports and one engineering-smoke report from the earlier development panel.

The [protocol](evidence_availability_protocol.md) and `configs/evidence_availability.json` bind the literal templates, source manifests, preparation/audit metadata, model revision and inference code. The inference digest incorporates the immutable v4 digest and the three v5 preparation/request/execution files. Preserve their bytes. Preparation timestamps are provenance, so a new reproduction will have a different preparation-metadata hash: create a new documented execution lock for that run rather than silently replacing the archived identity.

## Inference

Use the exact runtime in the archived `run.json`. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for local weights. CPU processor checks confirm equal native/degraded token IDs, equal overview tensors, distinct regional tensors after BF16 conversion, and identical native regional tensors across overview budgets. They do not prove attention or capture production activations.

```text
python experiments/evidence_availability.py --manifest data/tatdqa_v5/smoke_manifest.jsonl --model-dir models/qwen3-vl-4b --config configs/evidence_availability.json --output local-runs/evidence-smoke --smoke
python experiments/evidence_availability.py --manifest data/tatdqa_v5/main_manifest.jsonl --model-dir models/qwen3-vl-4b --config configs/evidence_availability.json --output local-runs/evidence-main
```

There are ten independent fresh requests per report: direct/native/degraded at 256, 512 and 1024 overview-token caps, plus a 4096-token full-page baseline. All ten are shuffled deterministically within report. Regional requests do not depend on a previous direct answer. One native region is resized once by the fixed sizing rule; its exact pixels are unchanged across overview budgets. Each degraded region is reconstructed from its corresponding overview into the same output dimensions.

Every action is flushed, every completed report is synchronized to disk, and completed files are hashed. Resume checks the exact identity, saved input geometry/pixels/messages and scores before continuing. Failed calls are recorded and stop execution; there are no automatic retries. Do not change prompts or eligibility after inspecting smoke accuracy. The smoke is an engineering check, not a tuning set.

## Analysis and reporting

```text
python experiments/analyze_evidence_availability.py --run local-runs/evidence-main --manifest data/tatdqa_v5/main_manifest.jsonl --output local-runs/evidence-analysis
```

The analyzer requires a complete run and validates all request, source, ROI, completion and scoring bindings. The sole primary contrast is native minus degraded at the 256-token overview cap, measured with the unchanged official single-span EM convention. Conservative text EM is mandatory secondary evidence. This prospective v5 choice differs from the strict-EM primary metric of v3/v4 and must not be presented as a direct improvement over their headline numbers. All invalid and truncated generations remain in denominators. Report-level paired bootstrap intervals retain all ten responses together.

Primary latency is each standalone invocation, including image decoding/construction/hashes, processing/transfers, generation/decoding and token-log-probability extraction. It excludes model loading, warmup, rendering, grading, logging and a deployable localization algorithm. The alternative decision-state scenario adds the matched direct invocation to the regional branch; sequential memory is the maximum, not the sum. Active-desktop timing is not a stable throughput guarantee.

The quantitative screen never authorizes controller training automatically. A separate condition-blind review inspects every primary-pair parsed-answer change, including pairs with equal grades. Record content repairs separately from formatting, equivalent spans, both-wrong and ambiguous cases. Resolve ambiguous cases conservatively and never modify official primary scores from this review. Only a positive quantitative screen plus at least three distinct unambiguous native-only semantic repairs and positive net semantic repairs can motivate a prospective test on untouched reports with a nonprivileged selector.
