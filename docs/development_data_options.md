# Options for a future detail-sensitive development panel

Verified **2026-09-25**, using creator websites, papers, and repository metadata. This historical assessment covers source documentation and metadata; archive-level eligibility and inference were outside its scope. These are candidates for **100–200 development examples**, conditional on the completed direction-control experiment; they are not a new evaluation set. V*Bench, HR-Bench, and TextVQA remain reserved.

## 1. Single-document DocVQA, training split

**Best initial fit for localized reading.** The task uses document-page images and manually written questions whose answers are a single text span. The release specifies image files, training JSON, alternate accepted answers, source-document IDs, and OCR text/boxes. Use released page images, not screenshots or enlarged thumbnails. Their native size distribution and whether 100–200 suitable high-resolution pages exist have **not** been verified. [Task description](https://site.docvqa.org/datasets/docvqa), [release schema](https://site.docvqa.org/datasets).

Score with ANLS and exact accuracy against accepted answers; preserve the benchmark's normalization. A single span does not guarantee a unique referent: repeated fields, signatures, and unclear row/column references need human screening. [Original paper, §5.1](https://arxiv.org/html/2007.00398v2).

**Access/terms:** the creators require login to the [RRC download portal](https://rrc.cvc.uab.es/?ch=17&com=downloads) and acceptance of its download terms. The portal could not be retrieved during this check. Documentation is verified; authenticated download availability and exact current license terms are **unverified**. Mirror license tags are insufficient evidence.

## 2. InfographicVQA, training split

**Useful for dense text plus layout**, with released infographic images and existing QA annotations. The official access route is the same RRC challenge portal, selecting “Infographics VQA”; registration/download limitations above apply. Native dimensions and eligible-panel size remain unverified. [Creator dataset page](https://site.docvqa.org/datasets/infographicvqa).

Answers can involve image spans, multiple spans, or arithmetic. Use ANLS plus exact accuracy, not GQA's scorer. Fine-grained evidence/operation annotations and additional-answer collection were performed for validation/test; do not assume these fields exist in training. Screen training questions manually for a unique visible reference and localized evidence. Global counting/layout questions would mix detail recovery with other reasoning demands. [Paper, Appendices A and C](https://arxiv.org/html/2104.12756v2).

**Terms:** Appendix A.3 states CC-BY for QA annotations and an image ZIP for research/educational use, with original URLs for other uses. This is the authors' release statement; current portal terms were not verified.

## 3. ChartQA, human-authored training subset

**Easier publicly visible access, weaker high-resolution assurance.** The author repository provides training PNGs, QA JSON, tables, and sometimes noisy element annotations. Its linked [author-hosted archive](https://huggingface.co/datasets/ahmed-masry/ChartQA/tree/main) is publicly listed as 875 MB; no login requirement appeared, but download was not attempted. [Repository/schema](https://github.com/vis-nlp/ChartQA).

Select simple retrieval questions, with tables used only for audit—not model input. Arithmetic and ambiguous visual references require exclusion under a declared rule. Official scoring allows 5% numeric error and otherwise exact matching. The author-hosted dataset declares GPL-3.0; the paper also discusses source-specific chart terms, so that tag alone does not settle image redistribution. [Dataset card](https://huggingface.co/datasets/ahmed-masry/ChartQA), [paper §5.1 and Ethical Considerations](https://aclanthology.org/2022.findings-acl.177.pdf).

## Conditional selection rule

At this planning stage, DocVQA was the preferred option subject to access and terms verification; the later [source-detail study](native_detail_data_sources.md) used TAT-DQA. Before inference, freeze image-size and legibility criteria, audit unique referents, select one question per source document, and record all exclusions. Count genuinely eligible originals before committing to 100–200; upsampling is not new detail. Keep source groups and duplicate images together. ChartQA is a fallback only if native-size screening succeeds; InfographicVQA is a subsequent layout-rich option. Pretraining exposure is unknown for all three. A curated training subset supports development diagnostics, not uncontaminated generalization or representative benchmark claims.
