# LookAgain v3: document detail intervention

This is an internally locked development experiment, not a third-party preregistration or a held-out transfer evaluation. The configuration records the lock time and exact manifests before GPU inference. The primary question is whether a regional view from a fixed-DPI source PDF rendering helps more than the same region reconstructed solely from the overview already shown.

## Population and selection

Use only the authors' TAT-DQA **training** release. Annotation SHA256: `3025b4ca9c3c87f5ecd2c504371e1ee92929ef1f31d8a3ac38bc3a5376582fab`; document ZIP SHA256: `412dde804ec9d5888b888a9d7179f2157bc2679b92379f9804e6f2062e7237d9`. Released 224x224 PNG thumbnails are unsuitable; render original PDFs at **200 DPI** with the recorded Poppler binary. Do not upscale the released thumbnails.

The main panel has **100 distinct original reports, one question/page per report**. Reserve two further distinct reports for engineering smoke checks; they never enter main estimates. Original report identity is `doc.source`, not the supplied PDF segment UID. `doc.page` is starting-page metadata, not a page count. Eligibility requires exactly one actual PDF page and one OCR page, answer_type=span, exactly one nonempty answer string, empty scale, req_comparison=false, at most 160 answer characters and 25 whitespace-delimited words, at least 100 PDF-extracted text characters, and rendered area >=2,097,152 pixels with both dimensions >=32. Text availability screens out image-only PDFs but does not prove that every glyph has vector fidelity. Record PDF/render hashes and dimensions. Answers need not be numeric-free and may contain units.

Order reports and eligible questions deterministically by SHA256 of seed `20260925` and the corresponding identifier. Selection uses structure and technical eligibility, never model correctness, confidence, predicted difficulty, crop success or answer position. The preparation script records exact ordering, exclusions and source provenance. PDF text, OCR, answer mappings and evidence are audit-only; they do not enter prompts, crop locations or model features. Annotation issues discovered in audits are reported; they do not silently change labels or remove model failures. This population is financial-document extraction, not generic natural-image VQA. One report per observation reduces repeated-document dependence; related companies/templates may still be correlated.

## Model, responses and actions

Frozen local Qwen3-VL-4B-Instruct, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16, SDPA, greedy generation, 64 new tokens per action, assistant prefill `ANSWER:`. Preserve the original v1 source package byte-for-byte. System instruction: `You answer questions accurately based on the provided images.` Every question ends with: `Answer by copying the relevant single text span from the page, including any units. Do not explain.`

| Action | Input | Calls/page |
| --- | --- | ---: |
| direct | Full-page overview, at most 1,024 visual tokens | 1 |
| highres | Standalone full page, at most 4,096 visual tokens | 1 |
| repeat | Direct history plus exact overview copy | 1 |
| native_tl/tr/bl/br | Direct history plus source-render region, at most 1,024 tokens | 4 |
| degraded_tl/tr/bl/br | Same history, region reconstructed only from the overview, same output size as native | 4 |

All follow-ups independently see the same fresh direct response, never another follow-up response. Highres is standalone and receives no direct answer. Four normalized regions are (0,0,.6,.6), (.4,0,1,.6), (0,.4,.6,1), (.4,.4,1,1). Regions are fixed and not question-conditioned. Native/degraded use identical text: `The additional image is an enlarged region of the original image. Reconsider your answer using the available visual evidence. ` followed by the common answer instruction. Repeat truthfully says `The additional image repeats the original image. Reconsider your answer using the available visual evidence. ` followed by that instruction. Region direction words are absent.

`Native` is a code label for a **200-DPI PDF rendering**, not a claim about native camera pixels. Overview/highres/native output sizes stay within source dimensions and the 32-pixel grid; token values are ceilings, not guaranteed equality between different actions. Native integer boxes use floor-left/top and ceil-right/bottom. Project that same box into overview coordinates without independent integer rounding. Degraded uses PIL BICUBIC resize with that floating-point box to exactly the native output dimensions. It never reads original ROI pixels. All follow-up overview hashes match direct; repeat second image matches exactly; each native/degraded pair has identical prompts, history, output dimensions and realized processor grids/input tokens. Two-stage filtering is part of this fidelity intervention; this is not a perfect frequency decomposition or proof of an internal attention mechanism.

## Evaluation and estimands

**Sole primary outcome:** conservative text exact match, lowercased with whitespace collapsed but punctuation, articles and numeric formatting retained. Parse the entire answer using the existing fixed ANSWER-marker parser; invalid/empty answers score zero. Never use target-dependent parsing, substring success or a semantic judge.

Secondary official EM/F1 use a narrowly scoped, attributed implementation checked against pinned author code at Doc2SoarGraph commit `71a02715849942b79c1f74934a139e4b56fc6826`. Both prediction scale and eligible gold scale are empty; never inject gold scale into predictions. TAT answer arrays represent required spans, not alternative acceptable references. Diagnostic ANLS lowercases/outer-strips strings, preserves internal spaces and uses normalized character edit distance strictly <0.5. ANLS is not an official TAT-DQA score. All metric differences remain visible.

For each report/page average the four quadrant scores, then average reports. The primary contrast is **native mean minus degraded mean**. A four-quadrant mean represents a uniformly chosen single region in expectation, not a policy that acquires all four regions. Secondary contrasts: native-direct, native-repeat, degraded-repeat, highres-direct, native-highres. Report all eleven cells, all metrics, invalid/truncated responses and fixes/harms against direct. Keep all failures in denominators.

Use 10,000 paired bootstrap resamples of 100 source reports, seed 20260925, retaining all actions per report. Report 95% percentile intervals as exploratory/unadjusted. Do not treat 400 region presentations as 400 independent documents. Analyze only the complete locked main panel. No interim quality inspection or sample-size extension based on outcomes.

Privileged upper bounds: best score over {direct,repeat,native4}, over {direct,repeat,degraded4}, and over their union. Compare the two matched six-action candidate sets. These use labels after the fact and are **not achieved controller performance**. Report gains beyond direct/repeat and beyond highres; no best-quadrant selection is a deployable result.

## Measurement and decision

Time synchronized full calls including PNG decoding, required resizing, RGB hashes, processor/transfers/generation/decoding/logprob extraction. Exclude dataset download, PDF rasterization, model load, warmup and JSONL logging; record preprocessing separately. A follow-up's deployment cost is fresh direct plus that branch; memory is the maximum allocated peak across both. Highres/direct cost one standalone call. Report mean/median/p95 latency, visual/input/generated tokens, allocated/reserved memory and total study runtime. No cross-call KV reuse, energy claim or equal-compute inference. Shuffle ten alternatives per page after direct with a stable per-page seed. Six two-token synthetic warmups span landscape/square/portrait direct/native views, without exhaustive shape coverage. No timing repeat is part of this protocol.

The engineering gate requires complete coverage and every pixel/prompt/grid invariant. Practical threshold **delta=2 percentage points** is fixed, not a power guarantee:

- Proceed to a larger untouched adaptive-selection study only if primary native-degraded mean >=delta with CI lower bound >0, and the matched native oracle exceeds degraded oracle by >=delta. This establishes possible headroom, not learnability.
- Revise if detail helps but fixed crops harm direct, oracle headroom is weak, or full-page highres provides the simpler quality/cost solution.
- Stop this particular setup if upper CI bounds for both primary and matched-oracle contrast fall below delta. Wide/mixed intervals are inconclusive, not evidence of no effect.

The decision combines all prespecified criteria. Controller training on this panel would require a separate report-level evaluation; resubstitution accuracy would remain a development estimate. V*, HRBench and TextVQA outcomes remain reserved. The pilot addresses the matched-detail question within this source population, with method novelty and broader transfer outside its scope.

## Source-audit sensitivity amendment, before main inference

The source-only audit found incorrect official references, including a neighboring-column value and a formula for tax attached to a profit question. Before any main-panel model outcomes are generated, freeze a per-question quality mask and its hash. The full 100-report primary analysis and original answers remain unchanged. Add a secondary sensitivity analysis that excludes only confirmed `label_error` and irreducibly `ambiguous` items; retain mapping-only issues and punctuation/formatting differences. Confirm questionable cases against full-page text or the rendered page, rather than treating a missing OCR substring as proof of an erroneous label. Publish all flags, reviewed IDs, reasons and visual-check coverage. This descriptive source review does not constitute independent expert validation. The two excluded smoke reports may be used for runtime/format checks; no main answer is inspected before the mask lock. The final configuration records the source-audit lock separately from the initial protocol lock.
