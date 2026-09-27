# Extraction and region ranker: prospective development protocol

This revision is a development experiment on all 124 sources already exposed by
the previous region study (64 development + 60 evaluation). The former evaluation
sources are now development data. Two old engineering sources remain excluded
from training. No new held-out, GQA, transfer or full-document evaluation is claimed.
The source page remains selected using dataset answer-page annotations; ROI
annotations and reference answers are unavailable to inference.

## Two fixed changes

1. Exactly one minimal-span extraction instruction in `refinement_core.py`, frozen
   before any new answers. Everything else in the answer interface is unchanged:
   source raster, nine half-width/half-height candidate windows (at most 25% page
   area), 256 overview / 1024 crop / 4096 high-resolution visual budgets, greedy
   decoding, `ANSWER:` prefix, 64-token cap, parser, exact-match references and
   secondary official scalar ANLS scorer. The direct and high-resolution controls
   receive the same instruction. No prompt search or post-hoc grade repair.
2. A small question-conditioned ridge region ranker trained only on frozen OLD
   nine-window exact-match grades. Before any main new answers, extract low-view
   features and commit every out-of-fold prediction and coefficient. No new-answer
   labels enter training, tuning, scaling or policy decisions.

## Features and training

Use the exact old low-resolution selector input, including its question and
candidate instructions, without generation or REGION/ANSWER prefill. One frozen
Qwen3-VL-4B decoder forward returns final normalized hidden states. Pool image
tokens whose merged-grid centers fall inside each candidate. Image tokens occur
before the question, so their causal states cannot encode the later question.
The query vector is the final nonpadding PROMPT token (after the instructions and
assistant header), not the literal final question token. No native crop is used.
Spatially pooled states are contextualized by the visual encoder and causal
decoder; they are not isolated local-pixel descriptors. Query interaction is an
ablation of this feature representation, not proof of causal question grounding.

Apply a fixed NumPy Gaussian projection 2560→32, seed 20260927, then L2 normalize
each region/query vector. Features are `[region32, region32*query32, position9]`.
Raw means use float64 accumulation, raw vectors float32, transforms and fitting
float64. Raw vectors are saved as derived-feature NPZ artifacts; projected vectors
are saved in JSON. CPU replay checks the projection from the saved raw vectors.
Artifact serialization is outside the per-action inference timer. Reject
zero/nonfinite vectors. Fit full73, image+position41, position9
ablations; center features and binary labels within each source; RMS scales use
training sources only, floor 1e-8. Optimize mean squared residual across training
rows plus `1.0 * ||weights||²`, no intercept or hyperparameter search. Ties choose
the lowest region ID. All sources remain included even if nine grades are equal.

Five folds: sort SHA256(`refinement-fold:20260927:` + source_cluster_id), assign
round-robin. Each prediction uses only the other four folds. Best-fixed-region
baseline likewise learns from its training folds. Center=5, deterministic random,
and the archived prompted-selector IDs are fixed baselines. Full-data coefficients
may be exported for deployment but have no independent performance estimate.

## Execution order and accounting

Reserve source/label byte hashes and old raw runs, perform actual CPU preprocessing
for all 21 possible requests per source, then lock code/config/protocol/runtime,
13 model/processor file hashes and input captures before GPU execution. Engineering
generation and feature checks precede main features; complete main features precede
OOF fitting and prediction commitment; that commitment precedes main new answers.

Per source, generation executes a fresh legacy selector, an answer at its CURRENT
replayed selected region, then the 11 new actions (direct, highres, nine windows) in
source-hashed order. The 2×2 scientific comparison always uses the ARCHIVED selector
ID and old archived answer bank; replay answers are technical checks only. Record
all replay discrepancies without silently switching the fixed policy or retrying.
If IDs disagree, fresh selector latency is explicitly a proxy for the archived
policy; quality still uses its fixed archived ID. Same ranker IDs apply to old/new
answer banks. Historical old costs remain separate from current costs.

Total: 126×13=1638 experimental generations and 126 feature forwards. Each of two
generation processes performs six synthetic-white, two-token warmup generations;
each of two feature processes performs three synthetic-white feature warmups.
Thus planned 1650 generation calls plus 132 decoder-only feature forwards. Any
failed attempts are retained and added to this ledger; no unreported retries.

Source decoding, request construction, CPU preprocessing and input hashing,
GPU forward/generation, CPU pooling/projection and synchronization are timed.
Per-source ranker prediction time is measured separately. Full/image+position
policies pay the feature forward; position/fixed/random/center policies do not.
Sequential GPU memory is the maximum of feature and chosen answer peaks, not
their sum. Uniform-window expectation averages nine answer costs plus one random
selection cost; its latency distribution is the equally weighted 9N mixture.
Model load, synthetic warmups and offline training are reported separately.

## Analysis decided before the new outcomes

Primary DEVELOPMENT diagnostic: full-ranker/new minus archived-prompted/new EM,
source-weighted out-of-fold mean and five fold effects. Overlapping cross-fit
training induces dependence: no naive paired-bootstrap confidence interval for
learned-policy comparisons. Report all sources, invalid parses, truncations,
gains/harms, ablations, training-only best-fixed, uniform/random/center, direct,
high-resolution, oracle (upper bound only), timing and memory. Sources are the
independent grouping unit, never nine windows treated as nine independent sources.

Fixed archived prompted policy new-minus-old extraction effect may use a paired
source bootstrap (10,000, seed20260927), explicitly conditional on reused data and
one development prompt choice. Report old/new prompted versus old/new OOF-ranker
2×2 means and descriptive interaction. This is neither confirmatory evidence nor
proof of transfer, novelty or a gain/stopping controller. A bounded deterministic
post-run case inspection explains format/content changes without altering grades.

The next research gate remains a fresh untouched source split and eventual
cross-task/model transfer, only if development results justify the added cost.
