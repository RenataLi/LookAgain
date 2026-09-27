# A small region ranker with a frozen VLM

The implemented component chooses **where** to look among nine fixed windows.
It does not yet decide **whether** another look is worth its cost. Every chosen
window is answered in a fresh request with the same full-page overview.

```mermaid
flowchart LR
    I[Provided page + question] --> O[256-token-budget overview]
    O --> F[Frozen Qwen3-VL decoder: one forward]
    F --> R[Nine spatially pooled region vectors]
    F --> Q[Final prompt-token vector]
    R --> P[Fixed projection to 32 dimensions]
    Q --> P
    P --> X[Region + region × query + position]
    X --> M[73-weight ridge ranker]
    M --> C[Commit one of nine region IDs]
    C --> A[Overview + native crop: fresh answer]
```

The query vector summarizes the entire prompt, including its image, question,
fixed candidate instructions and assistant header. It is not a standalone
question embedding. Region states occur earlier than the question in the causal
decoder and cannot see it, but the visual encoder already contextualizes them
across the overview. Their spatial coordinates do not make them isolated local
pixel descriptors. Multiplicative region/query features provide one simple way
to condition the scoring on the later query representation.

For source `s` and region `r`, the feature vector is
`x[s,r] = [v[s,r], v[s,r] * q[s], one_hot(r)]`, where `v` and `q` have unit L2
norm after a fixed 2560-to-32 random projection. The scorer centers features
within each source, applies training-only RMS scaling and uses a linear score.
Scores are ranking values, not calibrated probabilities or estimates of the
benefit of another observation. Exact floating-point argmax ties select the
lowest ID.

Training centers the nine binary old exact-match labels within each source and
minimizes mean row-level squared error plus `1.0 * ||w||²`. All 124 sources are
retained. Only 49 have different old grades among their windows; the 75 flat-label
sources still enter the normalization and objective. Each five-fold model sees
only 37–42 nonconstant training sources. This is a deliberately small baseline,
not sufficient evidence for a complex neural controller.

Three predeclared variants share the same fitting rule:

| Variant | Features | Per-query visual feature forward charged? |
|---|---:|---|
| Full | Region32 + interaction32 + position9 | Yes |
| Image + position | Region32 + position9 | Yes |
| Position only | Position9 | No |

Best-fixed-region is also learned only on the training folds. It chooses the
highest exact training count, breaking count ties by lowest ID. A numerical roundoff discrepancy between position-only and best-fixed in fold0
was recorded during training; it is not a substantive difference in learned
preference. The [research note](../reports/research_note.md) summarizes the results.

Five source-disjoint folds prevent each source's answers from fitting its own
scorer. They do **not** make this dataset untouched: all 124 sources were already
exposed during previous work. The same committed region IDs are applied to old
and new answer instructions, which makes their 2×2 comparison interpretable.
Only old labels enter fitting; the new answer bank is generated after predictions
are committed. [Full-data coefficients](../artifacts/ranker_coefficients.json) are included for
model inspection; they have no independent performance estimate.

Implementation: [feature extraction](../experiments/refinement_features.py),
[ridge fitting](../experiments/refinement_ranker.py),
[OOF training and commitment](../experiments/train_refinement.py), and
[execution protocol](refinement_protocol.md). The [numerical replay](reproduce.md) checks arithmetic from exported scores and
costs. Raw feature arrays, source text and original execution artifacts remain
outside this release; the historical strict path additionally requires them.
