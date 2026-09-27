# LookAgain: diagnostic pilot protocol

**Version:** 0.1, written 25 September 2026 before the planned model experiments.  
**Stage:** feasibility and action-effect diagnosis. No controller has been trained and no scientific outcome is claimed by this document. This is a versioned local protocol, not a third-party preregistration.

## Question and scope

When a frozen vision-language model has produced an initial answer, does a new view of a particular image region offer useful information beyond another answer attempt or additional text reasoning? If useful action differences exist, can a later controller predict them from information available **before** taking the action?

The eventual hypothesis concerns the transfer of action-value prediction relative to confidence-based intervention. This pilot cannot test that transfer hypothesis. Its purpose is to establish whether the proposed action space has enough complementary benefit to justify training a controller, and to measure the actual computation required. Adaptive cropping, value-of-information selection, and adaptive visual tool use already have close precedents; see [related work](related_work.md).

## Frozen model and execution contract

Use [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) with frozen weights. Resolve and record an immutable checkpoint revision, processor revision, package versions, device, dtype, attention implementation, seeds, prompts, decoding settings, image resizing rules, and output limits for every run. Model loading is reported separately from warm inference. Quantization is a different condition and must be recorded if used.

A hardware smoke run of **4–8 examples** checks loading, image handling, output parsing, peak memory, and timing. Its examples and measurements must be labeled **smoke only**. A subsequent 64-example engineering pass may estimate the cost of the full pilot. Neither sample is sufficient for the scientific decision gate below. Warm-up examples do not contribute to reported accuracy or timed steady-state summaries.

Before the main pilot, run a modest full-image resolution sweep on a separate development subset. Select a reasonable direct-answer setting using its accuracy/cost tradeoff and lock it. The direct-answer baseline must have sufficient resolution for a meaningful comparison with cropping. Retain a higher-resolution direct-answer condition as a standalone control. Record actual processed dimensions and visual-token counts: requested pixel budgets alone do not identify the effective input.

Initial engineering settings are BF16, SDPA, greedy decoding, and an `ANSWER:` final-answer marker. The output cap is 32 new tokens for direct answers, rechecks, and crops, and 128 for the text-reasoning branch. Initial image budgets are approximately 1,024 visual tokens for the base image and each crop, and 4,096 for high-resolution direct answering; these are starting settings, not a completed resolution selection. Record realized `image_grid_thw`, processed dimensions, and token counts. A crop branch includes the original base image as well as the crop.

## Data and split policy

The main diagnostic pilot targets **400 GQA question-image pairs used for project development**; a fixed panel of 300–500 may be used if source availability requires it. Freeze its exact size, IDs, source revision, image hashes, selection seed, and exclusions before inspecting model outcomes. Do not stop early when results become favorable. If fewer than 300 are available, report an incomplete pilot, not a completed study. The starting panel is an accessible convenience sample from GQA training images, assigned to this project's development role. Record its provenance and availability bias; it must not be presented as the full benchmark or an official GQA test evaluation.

Select examples without consulting model correctness. Prefer one question per source image for the initial panel. If multiple questions share an image, keep them together in both splitting and uncertainty estimation. Report question types and image sizes to make sampling limitations visible.

For later training, development, and confirmation, split by **source image**, not question ID. Keep all questions, crops, resized copies, and alternate action outcomes of an image in one partition. Check source IDs and exact image hashes, then use a documented perceptual-similarity check and inspect candidate near duplicates across partitions. Record unresolved overlap and exclusions. Development subsets from a public validation split remain development data in this project; renaming them does not create an untouched test set.

V*Bench, HR-Bench, and TextVQA are reserved for future transfer evaluation. Do not tune prompts, thresholds, proposal rules, or controller hyperparameters on their outcomes. Before that evaluation, fix exact benchmark versions, splits, sample membership, overlap policy, answer scoring, controller weights, and the final analysis in a test-lock manifest. Benchmark familiarity or possible VLM pretraining contamination still limits claims about unseen data; image-disjoint project splits do not rule out pretraining exposure.

## One-step interventions

Generate the initial direct answer once. Every intervention starts independently from that same initial state; one intervention's answer must never become another intervention's input.

| Condition | Information and role |
| --- | --- |
| Stop (`direct`) | Return the initial answer; no second model call. |
| Recheck (`recheck`) | Ask for another answer after showing the same original image at the same resolution. No additional source detail is introduced. |
| Text reasoning (`think`) | Allow an additional text reasoning attempt with the initial visual context, but no new image or crop. Record its separate output-token limit. |
| Four fixed crops (`crop_tl`, `crop_tr`, `crop_bl`, `crop_br`) | Independently show one overlapping quadrant crop from the original image, alongside the original context, then answer. Each crop is a distinct admissible action. |
| High-resolution direct (`highres`) | Answer from a higher-resolution full image in a separate initial call. This is a standalone comparator, not an action available after the initial answer. |

The normalized quadrant boxes `(left, top, right, bottom)` are `(0, 0, 0.6, 0.6)`, `(0.4, 0, 1, 0.6)`, `(0, 0.4, 0.6, 1)`, and `(0.4, 0.4, 1, 1)`. Pixel conversion is fixed by the run configuration and recorded with each outcome. Use identical proposals for all future policy comparisons. Crops must come from the original image bytes, not a previously downsampled display image. Record both crop coordinates and post-processor dimensions. Cropping may amplify detail but can also remove relational context; preserve the global view and evaluate harm as well as benefit.

Keep answer instructions and extraction rules consistent across branches where possible. Record any unavoidable prompt or budget differences. A crop improvement alone does not establish that new detail caused it: rechecking, extra computation, changed framing, and higher direct resolution are competing explanations. This pilot provides controlled diagnostic evidence, not a complete causal identification of perception.

## Scores, cost, and uncertainty

Use a deterministic, documented short-answer extraction and normalization for GQA. Preserve raw generations, parsed answers, reference answers, and parse failures. Until validated against the official evaluator, call this **pilot normalized exact match**, not official GQA benchmark accuracy. Do not use an LLM judge to repair unfavorable answers. Unresolved parsing failures count as failures in the primary all-example score; report their rate separately.

For every action, report accuracy and the paired change from Stop. Also report both transition counts and rates:

- **Gain:** initial answer wrong, action answer correct.
- **Harm:** initial answer correct, action answer wrong.
- **Unchanged:** correct-to-correct and wrong-to-wrong separately.

The net accuracy difference is gain rate minus harm rate when all examples have binary scores. Give conditional repair rates among initial errors and corruption rates among initial correct answers with their denominators. Show action effects by question type and image size descriptively; do not promote the best post-hoc slice to a confirmatory result.

Use paired bootstrap resampling at the **source-image level**, retaining all actions and questions within each resampled cluster. Use 10,000 resamples and a recorded seed for the main pilot report. Report 95% percentile intervals and the number of unique images; these intervals characterize this development panel, not a general population or a successful final hypothesis test.

Measure warm end-to-end wall time on the same hardware, with GPU synchronization around timed work. Include image decoding/preprocessing, crop construction, proposal generation, feature extraction, controller evaluation when present, and all required model calls. Report initial and incremental costs separately, and total policy cost as initial cost plus selected-action and decision overhead. Also record peak allocated/reserved GPU memory, input visual/text tokens, generated tokens, failures, and timeout/OOM handling. Model/data download and training-label generation are separate setup costs, never silently included in or removed from deployment comparisons. Equal token counts are not equal compute budgets.

Report mean latency and tail latency, accuracy versus measured latency, and memory under an explicitly stated execution mode. Use batch size one for per-request timing unless another mode is named. On a prespecified subset, repeat timed runs after warm-up and vary action order to expose cache or thermal effects. Do not infer speedups across different hardware or caching policies.

## Baselines and diagnostic ceilings

The pilot measures each fixed action and one uniform-random choice among the four crops, drawn once per example with the fixed seed. These establish action effects; they are not learned policies. A future confidence threshold and a learned selector must use the same proposals, pre-action information access, and cost accounting. Tune all thresholds on development data only.

An **admissible-action oracle** chooses the best observed result among Stop, Recheck, Text reasoning, and the four fixed crops for each example. It uses outcome labels and is therefore an optimistic diagnostic ceiling, not a deployable method. Report its action set and tie handling. Exclude the standalone high-resolution direct condition from this oracle. Computing all actions for offline diagnosis does not mean a deployed selector gets to inspect them for free.

A **ground-truth-box oracle**, if added later, uses annotated relevant objects to select regions. It has privileged information and potentially better proposals. Report it separately from the fixed-action oracle and from feasible policies. Separate proposal coverage, region localization, and answer improvement: a crop can overlap the correct object without helping the answer.

## Pilot decision, declared before outcomes

This is a descriptive decision gate, not a significance threshold or a promise of a positive result. Review the whole fixed panel once and publish gain, harm, cost, and failure evidence together.

- **Proceed to controller development** if fixed crops show useful repairs beyond rechecking/text reasoning, there is meaningful per-example variation among admissible actions, and reasonable direct resolution or the high-resolution control does not explain away the proposed benefit at comparable cost.
- **Revise the action space or question** if annotated case review suggests useful detail exists but fixed quadrants repeatedly miss it, lose necessary context, or incur excessive overhead. Any revision creates a new protocol version and a new development experiment.
- **Stop the learned-selector direction** if no useful headroom remains or the practical tradeoff is consistently dominated by the simpler controls. A documented negative diagnostic result is a valid output.

No choice here establishes transfer, novelty, or publishability. Log the decision and its evidence; do not keep changing the sample, prompt, or resolution while calling the result the original pilot. Retain failed runs and report deviations with dates and reasons.

## Conditional next stage: learned action value

Only after the pilot, construct training labels from paired action outcomes on training images. For an admissible action `a`, the target may be its answer-score change relative to Stop; the decision objective subtracts a declared cost penalty. Train a compact predictor of action value, then compare it with Stop, fixed/random crops, confidence gating, and an appropriate published-method comparator. Select the cost penalty on development data and report accuracy/cost curves instead of one favorable operating point.

Features must be available before the selected action: initial-image/question representations, initial answer statistics, and proposal geometry are possible inputs. Ground-truth answers may create supervised targets on training data. They must never enter inference features, proposal selection, confidence thresholds, or test-time decision making. Post-crop activations, corrected answers, and scores of unchosen actions are forbidden selector inputs. Fit scaling, feature selection, and calibration only on the training/development partitions.

Keep label-generation cost separate from deployment cost. Validate cross-model and cross-dataset transfer under a separately locked protocol, ideally including another VLM family. Neither controller training nor transfer evaluation is part of the current pilot artifact.

## Sources and reporting

Follow the [GQA documentation](https://cs.stanford.edu/people/dorarad/gqa/about.html), [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), and original benchmark licenses. Do not redistribute model weights or dataset images without the required rights. Report source revisions and instructions to obtain external assets.

The pilot release should contain the protocol and deviations, configuration and sample manifest, per-example predictions and costs where licensing permits, aggregate tables with uncertainty, selected success/failure examples, and a concise limitations section. Clearly distinguish implemented diagnostics, completed measurements, and planned methods. A runnable scaffold or a successful hardware smoke run is not a completed research result.
