# Planned controller: predict the value of another observation

**Status: prospective extension.** The implemented [region ranker](refinement_model.md) selects among nine fixed windows using frozen visual features. The controller proposed here would additionally decide whether another observation is worth its cost. It has not been trained or evaluated. The original 400-image pilot and the later ranking panel remain development data; this extension requires separate training trajectories and untouched source-level evaluation.

## Decision and target

After the initial answer, select Stop, Recheck, extra text reasoning, or one of the same four fixed overlapping crops. High-resolution direct answering remains a standalone comparator. Keeping candidate regions identical across selectors isolates action choice from proposal quality.

For example `i` and action `a`, define `delta(i,a) = correct(i,a) - correct(i,stop)`. Its values are +1 for repair, −1 for harm, and 0 for no accuracy change. Predict three probabilities with a shared action-conditioned head: `p(repair)`, `p(harm)`, and `p(unchanged)`. Expected gain is `g(a) = p(repair) - p(harm)`. This explicitly prices the risk of corrupting a correct answer; predicting only post-action correctness or initial uncertainty addresses a different target.

Choose the action maximizing `g(a) - lambda * estimated_incremental_seconds(a)`, with Stop assigned zero gain and zero incremental action cost. Stop when every other score is nonpositive; use a fixed tie rule. Controller overhead is paid even when it chooses Stop. Report it separately and include it in total policy latency. The cost estimate must exist before action execution; measured future generation length cannot enter selection.

## Features available before the action

Cache features during the frozen backbone's initial pass:

- Global mean-pooled visual features and region-pooled features for each candidate, obtained from the **initial overview's visual tokens**. Map token centers to normalized crop coordinates using the realized image grid. Do not encode a candidate crop at higher resolution before deciding to request it.
- A pooled frozen question representation, with question-token boundaries identified from the rendered prompt, excluding answers and labels.
- Initial-answer statistics: mean generated-token log probability, mean next-token entropy, generated length, and parse-validity flag. These are model observations, never reference-answer correctness.
- Candidate coordinates, area, action type, and expected token budget. Non-spatial actions receive the global representation and an action indicator.

Choose and document one feature layer/tap and pooling rule during development. Confirm the visual-token mapping and define deterministic handling of empty regions. Fit feature scaling only on training data. A compact shared MLP with two hidden layers (128 and 64 units) and a three-class output is the initial candidate; architecture and dimensions remain provisional until the feature interface is measured.

Feature hooks, probability extraction, pooling, memory transfers, and policy inference can change initial-pass runtime and memory. Benchmark the instrumented initial pass against the ordinary direct baseline. Count that difference and all decision/proposal overhead in policy curves; cached features are not free merely because the backbone is frozen.

## Training and evaluation separation

Generate every admissible action independently from the same initial answer on **new training images**. Full trajectory logs provide supervised outcome targets; reference answers may grade those targets. References, ground-truth boxes, post-action features, and unchosen-action outcomes must never enter inference features.

Split by source image before collecting features. Keep questions, crops, and all outcomes together; audit exact and near-duplicate images across partitions. Use a distinct development partition for architecture, regularization, calibration, cost estimation, and `lambda` selection. Lock features, weights, prompts, candidates, operating points, scoring, and analysis before final image-held-out evaluation. Keep transfer benchmarks untouched until a separately fixed transfer evaluation. Existing development results cannot become final-test evidence by relabeling them.

Fit the three-class predictor using cross-entropy without outcome-dependent resampling as the default. If weighting or resampling is introduced, document its effect on probability calibration. Estimate incremental action cost from training/development measurements and pre-action features, or start with development-only per-action means.

## Required comparisons and diagnostics

Compare the MLP with multinomial logistic regression on the same inputs, a question-only predictor, calibrated confidence gating, Stop, each fixed action, and seeded random cropping. Confidence thresholds use development data only. Keep candidate access and timing rules identical; show feature-ablation costs as well as accuracy.

For the routing comparison, hold the non-Stop action ranking fixed between confidence and gain gating; additionally report confidence gating with the development-best fixed crop. This distinguishes when-to-intervene effects from where-to-look effects.

Evaluate repair/harm probabilities with multiclass log loss, Brier score, and reliability plots; report expected-gain error separately. On held-out images, report gain, harm, accuracy, and paired image-bootstrap intervals along measured total-latency curves. Show all prespecified cost penalties and the development-selected operating point. An oracle using outcome labels supplies diagnostic headroom, not deployable performance.

A future evaluation of the proposed controller would include comparisons with [VOILA](https://arxiv.org/abs/2602.03007), [CropVLM](https://arxiv.org/abs/2511.19820), and [Beacon](https://arxiv.org/abs/2607.28595). The controller has not been trained or evaluated, and its methodological novelty remains unestablished.
