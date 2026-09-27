# Related work and positioning

Literature check: **25 September 2026**. This is a focused reading map, not an exhaustive systematic review. The summaries below are grounded in the linked primary paper records and official resources. No baseline reproduction or empirical comparison is claimed.

## What is already established

Adaptive zooming, external crop policies, value-of-information selection, and deciding when visual tools help are all active prior research topics. Combining a frozen VLM with a small learned component is not, by itself, a novelty claim. LookAgain begins with a bounded diagnostic pilot; a later contribution would need to be established through a specific method and credible comparisons.

| Work | Relevant overlap | Implication for LookAgain |
| --- | --- | --- |
| [V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs](https://arxiv.org/abs/2312.14135), Wu and Xie, 2023 / CVPR 2024 | Introduces guided visual search, the SEAL architecture, and V*Bench for demanding visual detail. | Visual search and repeated region inspection are established ideas. V*Bench is a candidate future transfer benchmark. |
| [Adaptive Chain-of-Focus Reasoning via Dynamic Visual Search and Zooming for Efficient VLMs](https://arxiv.org/abs/2505.15436), Zhang et al., 2025 | Trains Qwen2.5-VL to focus and zoom using supervised fine-tuning followed by reinforcement learning. | Compare the choice to keep the base VLM frozen; do not describe adaptive focusing itself as new. |
| [CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](https://arxiv.org/abs/2511.19820), Carvalho, Dias, and Martins, 2025; revised 2026 | Learns an external crop mechanism with reinforcement learning while leaving the target VLM unchanged; studies fine-grained perception and transfer. | A particularly close reference for the proposed frozen-VLM-plus-controller architecture and transfer claims. |
| [VOILA: Value-of-Information Guided Fidelity Selection for Cost-Aware Multimodal Question Answering](https://arxiv.org/abs/2602.03007), Bhope et al., 2026 | Predicts correctness by fidelity from question features, calibrates estimates, and selects fidelity using expected utility and cost before model execution. | Value-of-information and cost-aware fidelity selection are established. A post-answer spatial action-value study must explain its narrower distinction and compare relevant predictors. |
| [Beacon: Knowing When and How to Perform Agentic Visual Reasoning](https://arxiv.org/abs/2607.28595), Wang et al., 2026 | Studies whether tools are necessary and whether tool gains on difficult questions are offset by harm on previously solvable questions; trains adaptive behavior with reinforcement learning. | “When to use a visual tool” and gain-versus-harm analysis are close prior work. Measure both transitions and avoid asserting a new problem formulation. |
| [VisLens: Single-Pass Interpretable Visual Search for Multimodal LLMs](https://arxiv.org/abs/2608.30705), He, Kim, and Akata, 2026 | Uses a lightweight tuned lens to read visual-token semantics from early hidden states and localize relevant regions. | Learned lightweight localization and efficient visual search are close references. Feature extraction and proposal overhead must be included in comparisons. |

These papers use different models, information access, actions, training, and cost definitions. Their reported headline numbers cannot be compared directly to a small GQA development pilot. A named baseline implementation or detailed novelty claim requires review of the full methods, released code, checkpoints, and license conditions.

## A defensible question to investigate

The proposed direction is **prediction of the incremental value of a particular new observation after an initial answer**, with Stop, additional text reasoning, and repeat viewing as alternatives. A future study could test whether that prediction transfers more reliably than confidence gating under the same admissible regions and measured computational budget.

VOILA and Beacon cover related decision principles, CropVLM learns an external crop mechanism, and VisLens addresses lightweight localization. A distinct methodological contribution would require a specified difference in supervision, available features, action space or transfer protocol, supported by experimental comparisons.

The initial GQA pilot tested whether fixed region views offer complementary repairs after accounting for harm, extra attempts and direct resolution. Subsequent studies progressed to the implemented [region ranker](refinement_model.md). Expected-gain control and transfer remain separate research questions.

## Comparison policy

1. **Original-method reproduction:** use the authors' released implementation/checkpoint and evaluation contract where feasible; name the exact revision and disclose departures.
2. **Controlled reimplementation:** adapt a clearly specified component to this project's frozen model, fixed proposals, and budget. Label it “inspired by” or “controlled reimplementation of” the component. Do not call it a reproduction of the complete published method.
3. **Diagnostic controls:** Stop, Recheck, extra text reasoning, uniform-random/fixed crops, and high-resolution direct answering isolate parts of the mechanism. They do not substitute for a close published comparator in a later method paper.
4. **Privileged analyses:** fixed-action and ground-truth-box oracles are separate diagnostic ceilings. Neither is a feasible selector, and the latter has stronger information access.

A method comparison requires compatible releases and an explicit match of task, information access and budget. Controlled adaptations are reported with their departures from the original evaluation contract.

## Data and model resources

| Resource | Planned role | Primary sources |
| --- | --- | --- |
| GQA | Accessible development sample for the initial action-effect pilot; future image-disjoint controller development. | [Official description](https://cs.stanford.edu/people/dorarad/gqa/about.html) |
| V*Bench | Future transfer to questions requiring small visual details; keep outcomes untouched during initial development. | [Paper](https://arxiv.org/abs/2312.14135), [authors' repository](https://github.com/penghao-wu/vstar) |
| HR-Bench | Future transfer to high-resolution image perception. Its accompanying DC² method partitions images and aggregates patch descriptions, another relevant control family. | [Divide, Conquer and Combine](https://arxiv.org/abs/2408.15556), [authors' repository](https://github.com/DreamMr/HR-Bench) |
| TextVQA | Future transfer to reading and reasoning about text in images; use its own documented scoring rules. | [Official dataset](https://textvqa.org/), [download and split documentation](https://textvqa.org/dataset/), [paper](https://arxiv.org/abs/1904.08920) |
| Qwen3-VL-4B-Instruct | Frozen first VLM; pin both model and processor revisions before experiments. | [Official model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL) |

Benchmark documentation may be consulted while its outcomes remain reserved for transfer evaluation. Prompt and policy selection must exclude those outcomes. Reports must specify dataset versions, sampling, licensing, known overlaps, and possible pretraining exposure.

## Context for the direction diagnostic

Checked **25 September 2026**, independently of the ongoing direction-control outcomes. Three primary studies provide relevant context; none establishes the mechanism tested here or makes this diagnostic novel.

- **Controlled spatial relations:** Kamath, Hessel, and Chang's [What's "up" with vision-language models?](https://arxiv.org/pdf/2310.19785) (2023, §§2–3) tests photographs of the same object pairs in different spatial arrangements and captions differing by a preposition. The evaluated models struggle despite these controls. Its GQA-spatial construction also filters ambiguous object references. This motivates separating spatial recognition from object identification and reporting consistency across conditions. Its caption-selection results are not directly comparable to our free-answer, two-image follow-ups.
- **Reference-frame ambiguity:** Liu, Emerson, and Collier's [Visual Spatial Reasoning](https://aclanthology.org/2023.tacl-1.37.pdf) (TACL 2023, §§3.2, 4.3, 5.3) explicitly distinguishes viewer-relative and object-intrinsic frames. Its annotation protocol accepts a statement when true under either frame, illustrating why apparently conflicting left/right labels can depend on the interpretation. The paper also studies frame prediction and transfer to spatial judgments. This supports making the intended frame explicit. However, object viewpoint changes are different from changing an image's crop boundary; the paper does not demonstrate crop-induced coordinate confusion.
- **Prompt sensitivity:** Khemlani et al.'s [Vision language models are unreliable at trivial spatial cognition](https://arxiv.org/html/2504.16061v1) (2025 preprint, §3) evaluates three VLMs on TableTest with varied left/right prompts, object order, and response formats. Performance changes across logically related formulations. Crucially, their follow-up with an irrelevant relation does **not** support their initial hypothesis that models simply follow the first listed relation (§§3.3.4–3.3.5). Sensitivity alone does not identify a word-copying mechanism.

For LookAgain, these studies motivate the prespecified wording and frame controls, alongside the unchanged-pixel sham comparison. They do not predict its outcome. The sham falsely describes the overview as a crop, so named-versus-sham changes both pixels and image–text congruence. Directional answer shifts with fixed pixels would establish sensitivity to the supplied context in this setup, not prove literal cue copying, reference-frame confusion, or general spatial inability. No close crop-coordinate mechanism was verified in this focused reading; a broader novelty review remains necessary.

## Citation practice

References identify the primary versions consulted. Method reproductions and adaptations additionally require the released implementation revision and a description of any changes to models, inputs, training or evaluation. This focused bibliography is not an exhaustive prior-art review.
