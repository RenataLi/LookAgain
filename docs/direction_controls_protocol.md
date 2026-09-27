# Direction controls: development diagnostic v2

**Design rules locked: 25 September 2026, 17:17:18 UTC.** Exact prompts and analysis rules are recorded in [direction_controls.json](../configs/direction_controls.json), with the sham definition, per-image action shuffling and interpretation limits fixed and verified against the completed v1 metadata in `reports/pilot400/run.json`, before the first v2 smoke run or inspection of v2 outcomes. The runner records the executable and configuration hashes before generation. This is an internally versioned protocol, not third-party preregistration.

## Motivation and limits

The first fixed-crop pilot motivates a diagnostic of whether crop naming and coordinate-frame instructions change answers about directions. The present experiment reuses all 400 source images and questions from that development panel. It is not new data, a held-out confirmation, or a learned-controller evaluation. The nine crop-exclusive rescues in v1 form an exploratory, outcome-selected subset only; they never define the primary sample. This subset differs from the ten repairs available to the broader admissible-action oracle.

Keep the frozen VLM, weight revision, dtype, attention backend, image preprocessing, and generation settings fixed to the recorded v1 contract. The v2 implementation is separate from the unchanged v1 source. Recompute an initial answer for every image under the locked direct prompt. Do not substitute stored v1 answers or condition subsequent branches on v1 correctness.

## Crossed condition/quadrant panel

For every image, generate one fresh direct answer and sixteen independent follow-ups: four conditions crossed with the four original overlapping quadrants. All follow-ups begin from that image's same fresh direct-answer conversation. No branch receives another branch's answer.

This is not a full image-content × naming factorial design: a neutral sham condition is absent. Consequently, the experiment does not identify the interaction between image content and quadrant naming.

| Condition | Additional image | Follow-up instruction |
| --- | --- | --- |
| Named real crop | The actual quadrant crop | Explicitly names the quadrant. |
| Neutral real crop | The identical actual crop | Describes an enlarged region without naming its quadrant. |
| Named sham | A byte-identical copy of the first overview | Uses the identical named-crop instruction, despite showing the full image. |
| Named crop with original frame | The identical actual crop | Names the quadrant and explicitly asks for directions in the original image's coordinate frame. |

The quadrant rectangles remain `(0,0,0.6,0.6)`, `(0.4,0,1,0.6)`, `(0,0.4,0.6,1)`, and `(0.4,0.4,1,1)` in normalized left/top/right/bottom coordinates. Crop source bytes and interpolation are unchanged. The three real-crop conditions receive byte-identical crop images for each quadrant. The sham uses `overview.copy()` without another resize; its actual second-view box is `[0,0,1,1]` and its claimed crop box is the named quadrant. Record both boxes, realized image dimensions, and visual tokens. Rounding can produce different crop and overview shapes despite nominal 1,024-token budgets; no equal-compute claim follows. Each branch retains the original overview and adds one follow-up image. The named-real and named-sham prompts must be byte-identical for a given quadrant.

The sham deliberately gives an incorrect spatial description of its full-image input. It is a controlled counterfactual that changes image/text congruity as well as image content, not a pure wording control or a faithful region observation. The neutral-versus-named comparison changes language while holding the actual crop fixed. The frame-versus-named comparison adds a coordinate instruction while holding the crop fixed. These contrasts diagnose behavior and cannot, on their own, identify the model's internal reasoning mechanism.

There are exactly **17 planned calls per image and 6,800 calls overall**, excluding warm-up and documented infrastructure retries. Include every condition/quadrant pair for every image. Do not select the best-performing quadrant or retain only initially wrong examples for the primary analysis.

## Primary estimands and contrasts

Let `y(i,c,q)` be binary pilot normalized exact-match correctness for image `i`, condition `c`, and quadrant `q`. Let `d(i)` be the fresh direct answer's correctness. The primary condition score is `m(i,c) = mean_q y(i,c,q)` over all four quadrants, followed by an equal-image mean over the fixed 400-image panel. This estimates average intervention behavior, not the performance of a selector choosing a quadrant.

Report these three planned paired contrasts in percentage points:

1. Neutral real crop minus named real crop: the effect of removing quadrant naming for the same views.
2. Named crop with original frame minus named real crop: the effect of the explicit frame instruction.
3. Named real crop minus named sham: the effect of the actual crop under the same named instruction.

Also report every condition against fresh Direct, and all sixteen condition/quadrant cells descriptively. For each condition, average the wrong-to-right indicator and right-to-wrong indicator over quadrants, then over images; report both rates and their denominators. Their difference equals the condition-minus-Direct score change. Distinguish image counts from image-quadrant transition counts. Initial-correct and initial-wrong conditional rates are secondary analyses.

Use 10,000 paired source-image bootstrap draws with seed `20260925`. Each draw resamples image IDs and retains that image's direct result and all sixteen follow-ups. Recompute quadrant means and contrasts within each draw. Report 95% percentile intervals. Intervals are descriptive, unadjusted across contrasts, and do not constitute a familywise significance test. Do not promote a favorable exploratory contrast to a new primary endpoint after observing outcomes.

## Fixed direction subgroups

The primary analysis includes all 400 images. A prespecified secondary subgroup uses only the question text: Unicode casefold followed by the whole-word regular expression `\b(left|right|leftmost|rightmost)\b`. This literal rule includes hyphenated uses such as “right-hand” and excludes substrings such as “bright.” It may include semantically ambiguous uses of “right”; report that limitation instead of manually editing membership after viewing answers.

A separate secondary reference-label grouping marks examples whose conservatively normalized target answer is exactly `left` or `right`. These labels are for analysis and grading only; they never enter prompts, image construction, or generation. Report group sizes and overlaps. Existing question-type metadata may be summarized descriptively but cannot replace the locked lexical rule post hoc. Any direction-inversion count requires both the reference and parsed prediction to be exactly one of those two strings; report its eligible denominator.

## Execution, integrity, and reporting

Before execution, verify the exact prompts, quadrant names, output limits, manifest and model identities, source/configuration hashes, action order, scoring, subgroup rules, bootstrap seed, and error policy against the locked configuration. Verify the manifest contains the same 400 distinct source images as v1. The direct system message and question template are inherited unchanged from v1 `build_request`; all calls use greedy generation and a 32-token cap. A later rule change requires a new protocol/configuration version and an explicit deviation record.

Use the unchanged v1 conservative scorer: a bare one-line answer is accepted in its entirety; multiline output requires a unique final `ANSWER:` marker. Invalid parses count as incorrect. No automatic retries are allowed, including retries of incorrect answers. Execution errors fail loudly and are logged separately. Resume only under the same complete run identity, preserving completed records; log the interruption and resumption. Flush each record and synchronize files after each completed image. A completed primary report requires all 17 successful generation slots for every image; unresolved coverage must be labeled incomplete, never silently reduced to a favorable subset.

Measure each new call's full path and memory without changing preprocessing to improve a condition. Follow-up total cost includes the fresh direct call. Execute Direct first, then shuffle the sixteen follow-ups independently for each image using `random.Random(int(sha256(f'{seed}:{example_id}').hexdigest()[:16], 16))` on their configured canonical order. Preserve the observed order in records and disclose prompt-length differences. Warm-up uses representative 4:3, 1:1, and 3:4 inputs with Direct and one named true crop each, capped at two output tokens: six calls excluded from estimates. It does not exhaustively warm all realized shapes. Small source images may be upsampled; resizing creates no new source detail. Accuracy changes may reflect representation or attention effects rather than recovery of new pixels.

Use a distinct run directory, `direction400`, with `records.jsonl`, `run.json`, `summary.json`, and `report.md`; execution failures are recorded separately. Preserve complete per-example evidence and failures. Report v1-selected rescue cases only in a clearly labeled exploratory appendix. Final conclusions concern this reused development panel and these fixed interventions; transfer and controller usefulness require later independent experiments.
