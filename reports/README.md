# Research evidence

Eight experiments examine image detail, request wording, conversation history and
region selection in a frozen vision-language model.

Start with the [research note](research_note.md), [latest numerical data](benchmark/scores.json),
[expected results](benchmark/expected_summary.json), and [one-command replay](../docs/reproduce.md).

## Eight connected studies

| Study | Panel | Main generations | Experimental focus |
| --- | ---: | ---: | --- |
| Action-effect pilot | 400 GQA images | 3,200 | Direct answers, repeated views, text reasoning, fixed crops and high resolution |
| Direction and wording | 400 reused GQA images | 6,800 | Crop names, neutral language, reference frames and unchanged-pixel controls |
| Source detail | 100 TAT-DQA reports | 1,100 | Matched native-source and overview-derived crops |
| History and detail | 100 reused reports | 2,700 | Three conversation histories crossed with pixel fidelity and crop geometry |
| Evidence availability | 85 eligible reused reports | 850 | Overview budgets of 256, 512 and 1,024 visual tokens |
| DUDE detail replication | 660 document-source clusters | 2,640 | Same region and processor grid, different source fidelity |
| Prompted region selection | 60 evaluation clusters | 780 | Nine fixed candidates and a frozen VLM selector |
| Extraction and learned ranking | 124 reused development sources | 1,612 | Five-fold ridge ranking, feature ablations and answer extraction |

These panels contain **19,682 main model generations** in total. Engineering runs,
region-selector development runs and decoder-only feature forwards are accounted for
separately. Several studies deliberately reuse panels for controlled comparisons;
the row counts are not a count of unique independent documents.

## Selected observations

- **Crop wording.** On the direction-control panel, neutral wording improves
  mean crop EM by 8.69 percentage points over named crops, with paired interval
  [6.69, 10.81] pp. The unchanged-pixel controls help separate language effects from
  the visual contents of a crop.
- **Native versus reconstructed detail.** On 660 DUDE source clusters,
  source-native crops achieve 443/660 exact matches versus 424/660 for matched
  overview-derived crops: +2.879 pp, paired interval [0.758, 5.000] pp. Page and ROI
  selection are annotation-guided in this study.
- **Prompted selection.** The 60-source prompted selector achieves
  56.67% custom EM against a 54.07% uniform-crop expectation; the paired difference
  is +2.59 pp with interval [−2.78, +8.70]. Both quality and selector latency enter
  the subsequent learned-ranking design.
- **Learned ranking.** On 124
  reused development sources, the 73-feature ranker reaches 60.48% custom EM at
  0.765 s assembled mean latency. Prompted selection reaches 58.87% at 0.963 s;
  image+position features reach 60.48% at 0.756 s, and the high-resolution baseline
  reaches 81.45% at 1.571 s.

## Numerical release

`benchmark/scores.json` contains paired numerical scores, flags, fold membership and
measured cost observations. `benchmark/expected_summary.json` contains the expected
statistics. The standard-library replay independently recomputes their arithmetic.

The replay starts from recorded EM/ANLS scores. Scoring generated answers requires
the original strings and references; model execution also requires source media
obtained from the dataset providers.

`cohort_splits.json` is a numerical projection of historical cohort membership.
The test suite checks panel sizes, role separation and exclusion counts. The
production execution tests additionally exercise lock validation with synthetic inputs.

## Explore the implementation

| Topic | Entry point |
| --- | --- |
| Learned features and ridge objective | [Architecture](../docs/refinement_model.md) |
| Fixed latest-study choices | [Protocol](../docs/refinement_protocol.md) |
| Inference and experimental stages | [Execution guide](../docs/refinement_execution.md) |
| Region-policy evaluation | [Region-selection protocol](../docs/region_selection_protocol.md) |
| Matched-detail design | [DUDE replication protocol](../docs/dude_replication_protocol.md) |
| Literature and model/data resources | [Related work](../docs/related_work.md) |
