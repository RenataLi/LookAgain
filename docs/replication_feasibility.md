# Independent replication: feasibility decision

Historical planning note, 26 September 2026. At this stage no sixth inference run had started; the subsequent studies are summarized in the [research evidence index](../reports/README.md).

## Decision

The remaining local TAT-DQA training release cannot supply a well-powered, report-disjoint replication of a modest native-versus-degraded crop effect. Keep the four-condition comparison, but establish a larger independent source population before freezing and running the next confirmatory study. A small census of the remaining reports would be a separately labelled, low-power diagnostic; it would not fulfil the proposed confirmatory objective.

The source census established this population limit before model inference, cohort selection or protocol lock. Additional GPU capacity cannot increase the number of eligible independent reports in the release.

## What is actually available

| Stage | Original reports |
| --- | ---: |
| Local official training release | 172 |
| Previously used in main or engineering runs | 102 |
| Remaining before source eligibility | 70 |
| With eligible one-page OCR/question metadata | 65 |
| With at least one strict mapping/PDF/ROI candidate | 62 |

The 62 sources contain 944 technically eligible questions. Removing 36 generic questions such as “What does the table show?” leaves 908 questions from the same 62 reports. These are candidate counts, not a selected evaluation cohort. A new semantic source audit could reduce the pool further. Technical agreement between a reference and its OCR span does not prove that the question is unambiguous or that the selected row/year is correct.

All 102 previous original reports are excluded, including the two engineering reports and reports later excluded from v5. No pages or questions from those sources can re-enter under a different fragment ID. Filename normalization and exact-PDF checks found no additional cross-source matches; this does not prove absence of corporate aliases, shared templates or semantic duplicates.

The [TAT-DQA paper, section 3.3](https://arxiv.org/pdf/2207.11871.pdf) reports 182 original financial reports across all splits. Its 2,758 “documents” are page fragments from those reports. Therefore the published dataset scale cannot provide hundreds of untouched original reports after the 102 exclusions. The arithmetic upper bound of 80 remaining sources uses the paper's total; it is not a freshly computed union of downloaded train/dev/test annotations. Only the training assets were audited locally. The official fragment-level split does not establish original-report disjointness.

## Statistical target

The planning question is whether a paired study can detect a true **+5 percentage-point** correctness difference against zero with 80% probability. This does not mean proving that the true gain exceeds 5 points. Sensitivity calculations also cover gains of 3 and 3.5 points.

Power depends on the frequency of discordant paired outcomes: cases where exactly one crop condition is correct. The scenarios use 8%, 15% and 20% total discordance. An exact two-sided McNemar test at alpha .05 is enumerated over possible discordant counts and outcomes. The published calculations are prospective assumptions, not new model measurements. Shared-source dependence can reduce information beyond this independent-report calculation.

The [power-planning implementation](../experiments/plan_replication_power.py) computes these conditional scenarios. At the feasible 62-report ceiling, positive-effect detection power is **10.61%, 9.53% and 8.40%** under the three +5-point scenarios. Reaching 80% requires **264, 498 or 658** independent reports respectively; reaching 90% requires 343, 652 or 867. These requirements are conditional on the assumptions, not guaranteed for a new population. A nonsignificant result on the small available pool would remain inconclusive for modest effects.

One question per original report is the unit in this calculation. Several questions per report could support a different, report-clustered estimand, but would require a prospective covariance/power model. Multiplying the question count and applying an independent-pair test would be incorrect. Old development results must not be pooled with a new confirmation to reach the desired N.

## Contract to preserve in the next study

- Four independently generated requests: `direct_256`, `native_256`, `degraded_256`, `highres`.
- Frozen Qwen3-VL-4B-Instruct first; another VLM family is a later robustness study.
- Primary causal comparison: source-native versus overview-derived pixels for exactly the same reference-guided region, chat, image dimensions and processor grid.
- Reference-guided localization remains a privileged diagnostic. It does not validate a deployable region selector or controller.
- Keep the same overview/crop/full-page budgets and generation settings unless a source-only engineering check establishes a necessary change. Document and freeze any change before outcome collection.
- Fix the population, source exclusions, one-question selection rule, final N, metric, test, interval, semantic review rule and stopping rule before generation.
- Use a report-paired bootstrap alongside the exact test; do not choose the more favourable inference method after seeing results. Record invalid and truncated outputs without removing them from the denominator.
- Measure the full synchronized inference path. Distinguish standalone calls from a policy that first pays for a direct answer. Localization remains privileged and its cost is not measured by these calls.

These items describe a design to be finalized, **not a locked protocol**. No final cohort, configuration, run identity or completion artifact exists yet.

## Adapting to another dataset

A larger document dataset would be a new-domain replication of the controlled visual comparison, not a larger TAT-DQA sample. Before committing, verify creator-published access, source-document identities, original resolution, question/reference quality, answer-page/region alignment and sufficient eligible independent sources. Sample-size assumptions must be stated for that population.

The [source assessment](replication_data_options.md) and DUDE metadata census establish a credible route for further preparation. They do not establish a final eligible sample. In particular, one inspected author-supplied example has inconsistent coordinate systems across OCR providers and a questionable reference unit. Source geometry and semantic checks remain necessary even when an answer box is present.

Do not silently transfer the TAT-QA scorer to another benchmark. Its number/unit normalization is dataset-specific. A new dataset needs a prospectively declared correctness definition, with the official benchmark score reported in its appropriate role. The exact paired-binary power analysis applies to a binary correctness outcome; it does not establish power for ANLS or another continuous score.

Reserve a separate small engineering cohort before any main sample selection. Review source questions, references and mapped regions without VLM outputs. Freeze a ranked replacement rule or exclude reports prospectively; never replace a hard case because its generated answer is wrong. Check provenance and duplicates before downloading or rendering the full evaluation pool.

If a suitable pool cannot be established, the completed v1–v5 diagnostics and this feasibility assessment remain the available evidence. Without sufficient new independent data to establish the benefit of additional observations, these results do not justify controller training.

## Reproducibility

The original local planning archive retains source counts, per-source eligibility, a question-level audit, exact-power scenarios and verification records. The public [study index](../reports/README.md) summarizes the subsequent experiments. The new planning scripts read raw sources and previous source manifests; they do not read model outcome records. Earlier inference sources, protocols, raw outputs and archived releases remain frozen.
