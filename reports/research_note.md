# LookAgain: controlled visual re-observation and learned region ranking

## Research question

LookAgain tests how image detail and region choice affect a frozen vision-language
model's answers. The experiments control crop geometry, request wording,
conversation history and visual budget. The learned component ranks nine regions
using features from the frozen model.

## Method

The main backbone is Qwen3-VL-4B-Instruct, pinned to revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. The reference execution uses BF16, SDPA,
greedy decoding, a fixed answer prefix and a 64-token answer cap.

Each supplied answer page has nine geometry-only candidates. The windows overlap
on a 3×3 grid, each spanning half of the source width and height. A decoder-only
forward on a 256-visual-token-budget overview provides region and final-prompt
states. A fixed Gaussian projection reduces 2,560 dimensions to 32, followed by
L2 normalization.

For each candidate, the full model concatenates a 32-dimensional region vector,
32 elementwise region–query products, and a nine-way position code. A ridge model
with fixed regularization α=1 ranks the resulting 73-dimensional feature vectors.
Within-source centering is followed by RMS scaling learned from training folds.
Image+position and position-only variants provide feature ablations.

The selected native crop, with a 1,024-visual-token budget, enters a fresh request
alongside the overview. No evidence-box annotation, OCR transcript or reference
answer enters region selection. Dataset annotations supply the answer page.
The high-resolution baseline uses the full page with a 4,096-token visual budget.

![Architecture](../assets/method.svg)

## Development design

The latest panel combines 64 former development and 60 former evaluation sources,
making all 124 **reused development sources**. Five source folds have sizes
25/25/25/25/24. Rankers learn from the archived nine-crop answer grades; each
out-of-fold choice is committed before collecting answers under the revised
extraction instruction. The answer-instruction comparisons use the same chosen
region IDs. Two engineering sources are separate from this panel.

Custom exact match collapses case and whitespace while preserving punctuation,
units and numeric formatting. Scalar ANLS supplies a complementary string-similarity
measure. Invalid and truncated outputs stay in the denominators. A source remains
one paired statistical unit across policies and crops.

## Results

| Policy | Custom EM | ANLS | Mean assembled time |
| --- | ---: | ---: | ---: |
| Direct overview | 58.06% | 0.73671 | 0.473 s |
| Uniform expected crop | 55.29% | 0.68723 | 0.644 s |
| Prompted selector | 58.87% | 0.71731 | 0.963 s |
| Image + position ridge | 60.48% | 0.73386 | 0.756 s |
| Full ridge ranker | 60.48% | 0.72781 | 0.765 s |
| High-resolution page | 81.45% | 0.87989 | 1.571 s |

![Results](../assets/results.png)

The primary full-ranker minus prompted-selector contrast is +1.613 percentage
points: eight gains and six losses under custom EM. Full ranker assembled time is
20.57% lower than prompted selection. The image+position ablation achieves the
same exact-match count. Full-page high resolution achieves the highest custom EM
among the evaluated policies.

The five full-minus-prompted fold effects are 0, 0, +4.00, 0 and +4.17 pp. These
describe cross-fitted development results with shared training sets. The fixed
prompted-region extraction contrast changes 74 to 73 exact matches: −0.806 pp,
paired development interval [−4.839, +3.226]. Direct overview changes 62 to 72;
high resolution changes 99 to 101. This separates the answer interface from the
region-choice comparison.

Only 49/124 sources have varying archived grades across their nine candidate
crops. Every source is retained in fitting and analysis. The best-of-nine
label-aware diagnostic reaches 95/124 exact matches, identifying where candidate
coverage and selection interact. It is reported separately from executable policies.

## Compute accounting

The reference hardware is an RTX 5090. Policy latency assembles measured CPU
preprocessing, decoder features, ranking/selection and answer generation as needed.
These component measurements were collected on an active desktop; reported policy
latency is their sum. Full-ranker feature cost is charged to that policy;
position-only and fixed-region policies use their corresponding CPU choice costs.
Offline fitting, model loading and warmups have separate accounting.

The main generation bank contains 1,612 model generations, with 124 main feature
forwards. Source-level observations support the means, quantiles, paired contrasts
and fixed-policy bootstrap reproduced by the numerical release.

## Reproduce and extend

Run `python scripts/reproduce_results.py` from the repository root. The numerical
data retain source pairing, committed fold membership and measured costs. The
replay checks arithmetic from precomputed scores; experiment modules implement
scoring from generated responses and references. Source-document strings are
excluded from the numerical data.

The [model description](../docs/refinement_model.md) specifies the feature construction,
and [ranker coefficients](../artifacts/ranker_coefficients.json) provide the full-data
parameter sets fitted on all 124 sources. Reported quality estimates use the
out-of-fold predictions, not these full-data fits. The
[protocol](../docs/refinement_protocol.md) defines the controlled comparison, and the
[study index](README.md) connects it to the preceding visual-detail experiments.
