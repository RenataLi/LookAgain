# Research position of the source-detail control

**Prospective research rationale, recorded before native-detail outcomes.** The source-detail control tests a prerequisite for an action-gain policy: whether a higher-fidelity view can improve answers under matched geometry and prompts. The locked [protocol](native_detail_protocol.md) governs its estimates and decision thresholds. The completed study and subsequent experiments are summarized in the [study index](../reports/README.md).

## Why the paired control matters

A crop can change an answer without supplying additional source detail. It changes the presentation, repeats the question, adds a generation attempt, and can interact with the model's previous answer. Enlarging a region reconstructed from an already-shown overview cannot recover detail discarded when that overview was formed. Consequently, comparing a crop only with the initial answer cannot isolate the benefit of access to a higher-fidelity source.

The paired intervention presents either a region from the 200-DPI source PDF rendering or the corresponding region reconstructed solely from the overview. Within each pair, the question, initial answer, follow-up wording, region geometry, output size and realized model input grid are matched. Their difference therefore provides a necessary sanity check for claiming useful source detail in this setup. Repeat and standalone high-resolution answering provide complementary controls for another attempt and a larger full-page view.

This remains a specific fidelity intervention. Its two processing paths use different resampling histories; it is not an exact frequency decomposition or an identification of the model's attention mechanism. “Native” names the source-render branch in code, not native camera pixels. A positive contrast would show that this access route helps under the fixed protocol; it would not prove that a controller can predict the helpful cases, choose their regions, or outperform a simpler high-resolution call.

Financial-document extraction also differs substantially from GQA natural-scene questions. Dense text, table structure, units and reference transcription dominate this panel. Source-only label auditing and report-level grouping improve its interpretation, but do not make it evidence of natural-image transfer. This is a new development population, not an untouched final test of the earlier idea.

## Closest existing ideas

Three already-verified primary references delimit the proposed direction:

- [CropVLM](https://arxiv.org/abs/2511.19820) learns an external crop mechanism while retaining the target VLM. A frozen backbone plus a learned crop component is therefore not sufficient novelty.
- [VOILA](https://arxiv.org/abs/2602.03007) uses value-of-information estimates and cost to choose fidelity. A later LookAgain study would need to justify the value of a *post-answer, region-specific* gain predictor relative to a question-based fidelity decision, under compatible information and cost accounting.
- [Beacon](https://arxiv.org/abs/2607.28595) studies when visual tools help and when they damage answers. Recording repairs and harms, or deciding whether another view is needed, is not a new problem formulation by itself.

These are positioning references, not methods reproduced in this experiment. Any eventual named baseline must distinguish an author-code reproduction from a controlled adaptation with different models, proposals, training and budgets. The [related-work notes](related_work.md) provide the broader reading map.

## Outcome-dependent next steps, fixed before reading results

| Observed pattern | Actionable next step |
|---|---|
| Source detail clears the protocol's primary and matched-oracle gates, with useful cost headroom | Collect separate training trajectories on new reports; test whether pre-action features predict gain and harm. Passing these gates permits that study, not a claim of learnability. |
| Source detail helps, but full-page high resolution offers the simpler quality/cost tradeoff | Prioritize a direct fidelity-routing baseline. A regional controller needs an additional measured advantage to justify its overhead. |
| Some regions repair answers, but uniform crops cause substantial harm or only the privileged oracle looks strong | Treat selection as the unresolved problem. First test simple confidence, question-only and linear predictors on separate data; do not report oracle choices as policy performance. |
| Effects are small with tight intervals below the locked practical threshold | Stop this particular setting. A changed task, source-resolution regime or action space is a new study with a new protocol. |
| Intervals are wide, metrics disagree, or source-quality sensitivity changes the interpretation | Report the uncertainty and label limitations. Plan an independently sampled follow-up; do not extend this panel or tune thresholds in response to its outcomes. |

All follow-up costs include the initial answer plus the chosen branch. High-resolution direct answering costs one standalone call. A mean across four fixed regions represents uniform single-region selection in expectation; it does not hide the cost of acquiring four views.

## Requirements for evaluating an action-gain policy

The [controller proposal](controller_design.md) predicts expected incremental correctness, including harm, minus a measured cost penalty. Its inputs must exist before requesting the action: question features, initial-answer statistics and features from the initial overview. Higher-resolution crop encodings, reference answers and future action outcomes cannot be selection features. Feature extraction and decision overhead must be measured even when the policy stops.

An action-gain policy evaluation requires separate report-level training, development and locked test partitions; duplicate/template checks; enough examples to estimate policy improvements; calibrated gain probabilities; strong simple selectors and a close published comparator; and measured accuracy–latency curves against well-tuned direct resolution. Ablations must separate *when to intervene* from *where to look*. A second VLM family and reserved task/domain evaluations would test generality.

The target is a specified predictor, supervision rule or transfer property that improves a defined decision under a measured budget. Evaluation includes negative results, source-label limitations and comparisons with established methods.
