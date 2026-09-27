# Reproduce the history-context diagnostic

This guide records the original local experiment workflow. Its commands require the historical source manifests, raw records and identity artifacts, which are not bundled in this numerical release. See the [study index](../reports/README.md) and [current CPU replay](reproduce.md) for the published materials.

This v4 experiment reuses the exact 100 development reports studied in v3. Their outcomes motivated the intervention, so this is neither a held-out evaluation nor transfer evidence. No controller is trained. The frozen [protocol](history_context_protocol.md) defines all 27 calls per report and the interpretation limits.

## Data and environment

Follow [native-detail execution](native_detail_execution.md) to obtain the original TAT-DQA training release, render the selected PDF pages, and restore the unchanged main100/smoke2 manifests beside their `images/` folder. Do not use the release's 224x224 thumbnails. The archived report manifests alone do not supply image pixels. Model weights, PDF archives and full source PNGs are excluded from the source package.

Use the environment recorded in `reports/history100/run.json` and the pinned Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. The measured run uses local weights, BF16, SDPA, greedy decoding, 64 output tokens and the `ANSWER:` prefill. Model-generated token log probabilities are diagnostics, not calibrated correctness estimates.

`configs/history_context.json` binds both manifests, the earlier source-only quality mask, the original v3 experiment and the new inference code. The v4 digest depends on the immutable v3 source digest as well as `history_context.py` and `history_context_core.py`. Preserve exact file bytes; `.gitattributes` protects the byte-pinned configuration/metric files. Do not alter a locked protocol to resume an existing run. Create a new experiment identity for changed inference.

Run commands below from the repository root after installing the project dependencies. Substitute the matching prepared data and local model directories. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for offline inference.

## Engineering checks and measured inference

```text
python -m pytest tests -q
python experiments/audit_history_processor.py --manifest data/tatdqa_v3/smoke_manifest.jsonl --model-dir models/qwen3-vl-4b --output local-runs/history-processor.json
python experiments/history_context.py --manifest data/tatdqa_v3/smoke_manifest.jsonl --model-dir models/qwen3-vl-4b --config configs/history_context.json --output local-runs/history-smoke --smoke
python experiments/history_context.py --manifest data/tatdqa_v3/main_manifest.jsonl --model-dir models/qwen3-vl-4b --config configs/history_context.json --output local-runs/history-main
```

CPU processor checks inspect the serialized role template, two image blocks and actual processed tensors. They confirm identical pixels across history conditions and distinct native/degraded regional tensors, including after BF16 conversion. They do not capture production model activations or prove attention to either view.

Inference writes each raw record immediately, flushes every action and synchronizes the file after each report. It records a full identity in `run.json` and a record-file hash in `completed.json`. It rejects orphan records, mismatched identities, altered saved requests/scores, missing direct responses, unexpected actions and inconsistent completed-run markers. There are no automatic retries. A restart requires the same config, code, model identity, runtime and manifests; saved records are rebuilt and checked before continuing.

Every report runs fresh direct first, then a deterministic shuffle of the remaining 26 actions. Each branch is independent and re-prefills its full input; previous branch answers and cross-call KV caches are not reused. No reference answer enters a model request. Actual history includes the fresh direct response; placeholder uses the fixed acknowledgement; fresh includes no earlier assistant turn. The direct-response hash is recorded separately from the hash of inserted assistant content.

## Analysis and cost interpretation

The strict analyzer reconstructs all request geometries and pixel/message hashes, re-scores every raw answer and checks the completion/manifest/config/code bindings before aggregation. Use the prepared source manifest beside its matching images, not the provenance copy in `reports/history100/`.

```text
python experiments/analyze_history_context.py --run reports/history100 --manifest data/tatdqa_v3/main_manifest.jsonl --quality-mask reports/native100/source_quality_mask.json --v3-run reports/native100 --output local-runs/history-recomputed
```

Primary conservative EM keeps punctuation and numeric formatting. Official single-span EM/F1 use the same pinned implementation as v3; ANLS is diagnostic. Invalid and truncated answers remain in denominators. All 100 reports are primary, and the previously frozen 94-report subset is a separate sensitivity analysis. Confidence intervals resample whole reports, retaining all 27 responses together. The analysis includes every declared action and does not select a favorable region after scoring, except explicitly labeled privileged oracle diagnostics.

The primary estimand is the change in the native-minus-degraded advantage when the previous answer is replaced by the fixed acknowledgement. Fresh-context interactions and absolute detail effects are secondary. The practical advancement rule concerns an absolute detail effect, not simply a better fresh answer: it requires at least 2 percentage points, a positive paired interval lower endpoint and native accuracy at least as high as direct. It can only motivate testing untouched reports.

Per-invocation latency includes PNG decoding, view construction, hashes, processing, transfers, generation, decoding and log-probability extraction, synchronized at the GPU. PDF preparation, model loading, warmup, grading and logging are separate. For the primary decision-state scenario, every branch costs direct plus its own call. For the standalone scenario, fresh and placeholder cost their own call alone; actual and repeat still require direct plus follow-up. Direct/highres are standalone in both scenarios. A mean across four regions describes selecting one uniformly, not running four crops.

Latency is a measurement from one local run on an active desktop, not an isolated throughput benchmark. Allocated GPU memory and allocator-reserved memory are different quantities. The completion file's `elapsed_s` refers to the current execution segment after warmup; if resumed, it excludes earlier segments. `sum_measured_action_elapsed_s` covers recorded invocations across segments and excludes gaps between them.

## Reporting limits

Actual versus fresh changes the previous-answer content, turn structure, text placement and instruction repetition together. Placeholder preserves the layout but changes content and length; it is not token-matched or semantically inert. A positive interaction can result from degrading the control, so absolute performance and harms must accompany it. Agreement with direct does not prove copying. Zero empirical paired variability does not establish population equivalence or rule out rare gains.

The gallery's behavior-selected cases illustrate errors and repairs; they are not a representative sample or independent validation. The HTML is standalone with local assets. Static asset/content checks and visual inspection of saved inputs are separate from browser interaction testing.

Rebuild presentation artifacts only after the strict completed-run analysis:

```text
python experiments/render_history_context.py --run reports/history100 --manifest data/tatdqa_v3/main_manifest.jsonl --summary reports/history100/summary.json --quality-mask reports/native100/source_quality_mask.json --output local-gallery/history/gallery.html --plots local-gallery/history-figures
python experiments/build_history_context_note.py --summary reports/history100/summary.json --output local-gallery/history-note.pdf
```

Use a new gallery directory; the renderer refuses to overwrite prior figures or case assets. The PDF requires the optional report dependencies and the run-bound `interpretation.json` beside the summary. The interpretation must correspond to the supplied run; results from another run require a matching interpretation. The original three-page PDF was rendered and checked page by page. These outputs remain in the local research archive. The public [study index](../reports/README.md) summarizes the experiment; source-bearing records and page pixels are excluded from the numerical release.
