# Sources for the native-detail development stage

Source check: **25 September 2026**. TAT-DQA was selected for preparation of a 100–200-example development panel. Selection is conditional on the downloaded training metadata and PDF audit; this document does not establish the final sample size or image-quality distribution. No evaluation results inform these source notes.

## Official access and source structure

The [TAT-DQA creator repository](https://github.com/NExTplusplus/TAT-DQA) and [project website](https://nextplusplus.github.io/TAT-DQA/) link directly to this [public dataset folder](https://drive.google.com/drive/folders/1SGpZyRWqycMd_dZim1ygvWhl5KdJYDR2). An unauthenticated request returned HTTP 200 and its file listing. Training QA and document files are separately listed:

| File | Author-provided Drive file ID | Purpose |
| --- | --- | --- |
| `tatdqa_dataset_train.json` | `1FQUbZRlJKB-1sbvZ_HiZtE5isNwSvjki` | Training questions and existing annotations |
| `tatdqa_docs_train.zip` | `1VX4U0wx4ojITs72d5FAKvNJMkL5J6Xdw` | Training source PDFs and associated document content; listing size 1,172,553,879 bytes |

The release describes PDF documents of up to three pages, separate extracted/OCR content with word and block boxes, and QA records. Document fields include `uid`, source financial-report name, and starting page. Question fields include `uid`, `question`, `answer`, `answer_type`, `scale`, `req_comparison`, and evidence mappings. The published training set contains 13,251 questions over 2,207 documents, including 5,737 single-span questions; these are full-release statistics, not final-panel counts. [Dataset schema](https://nextplusplus.github.io/TAT-DQA/), [paper, §3 and Tables 1–2](https://arxiv.org/pdf/2207.11871).

Freeze source hashes, selected question/document IDs, source-report grouping, exclusion reasons, and rendering settings. One selected question per source report avoids treating multiple pages/questions from the same report as independent examples. Existing evidence annotations may support data-quality review; using them to select a model's crop would create a privileged action and must be declared separately.

## PDF resolution is not a native pixel count

A PDF page has physical dimensions and may contain vector text, raster images, or both. Rendering vector glyphs at fixed DPI preserves information from the original document; enlarging a low-resolution embedded scan does not add detail. A 2,000-pixel render alone cannot establish native image quality.

For each selected page, record page size, render DPI, output dimensions, source hash, and whether relevant content comes from vector text or a raster scan. Inspect effective scan resolution and legibility where applicable. Preserve an original-detail raster for crop extraction and derive any low-resolution overview from it. Describe this as a **rendered source-document detail experiment**, rather than claiming all source PDFs have a particular native pixel resolution. The website's example box `[0,0,1239,1754]` is only a schema example.

## Scoring contract: verify the implementation, not just its label

The creator-linked [Doc2SoarGraph repository](https://github.com/fengbinzhu/Doc2SoarGraph) supplies the evaluator. The inspected revision is **`71a02715849942b79c1f74934a139e4b56fc6826`**.

- [`tatqa_eval.py`](https://github.com/fengbinzhu/Doc2SoarGraph/blob/71a02715849942b79c1f74934a139e4b56fc6826/tatqa_eval.py) expects predictions keyed by question UID, each containing `[answer, predicted_scale]`. A missing prediction scores zero.
- [`tatqa_metric.py`](https://github.com/fengbinzhu/Doc2SoarGraph/blob/71a02715849942b79c1f74934a139e4b56fc6826/tatqa_metric.py) expects a list for a gold `span` answer. It sorts and joins answer elements; they are **not alternative acceptable answers**. Exact match compares normalized strings; span F1 uses token sets. Numeric strings incorporate their scale; an explicit `%` is handled without multiplying by the scale again. The defined `add_percent_pred` helper is not called in this revision. Number matching inside token-F1 alignment is disabled, so a partially matching numeric span can receive partial F1.
- [`tatqa_utils.py`](https://github.com/fengbinzhu/Doc2SoarGraph/blob/71a02715849942b79c1f74934a139e4b56fc6826/tatqa_utils.py) handles case, articles, punctuation, whitespace, numeric formatting, parenthesized negatives, and scale words. Percent maps to 0.01. These transformations differ from the GQA scorer.

For a one-field answer interface, declare the adapter's `predicted_scale=""`; never fill it from the gold label. A prespecified subset with one gold span and empty gold scale avoids requiring a separate scale prediction. Otherwise the answer must encode the appropriate quantity or unit. Isolated official helper checks confirm these examples:

| Gold answer / scale | One-field prediction; predicted scale empty | Exact match |
| --- | --- | --- |
| `12` / `million` | `12 million` | 1 |
| `12` / `million` | `12` | 0 |
| `12` / `percent` | `12%` or `0.12` | 1 |
| `12` / `percent` | `12` | 0 |
| `2018 revenue` / empty | `2019 revenue` | 0; span F1 is 0.5 |

These are synthetic formatting checks, not model results or a full benchmark reproduction. Use normalized exact match as the binary correctness measure; report token F1 separately if included. Validate the adapter against the pinned implementation before inference and preserve any declared departure.

## Terms and other datasets considered

The current TAT-DQA [repository](https://github.com/NExTplusplus/TAT-DQA#license) and website state **CC-BY 4.0**; the [earlier paper abstract](https://arxiv.org/pdf/2207.11871) says non-commercial use. Both statements are retained in provenance without a legal interpretation. Source images/PDFs are kept local for research; the repository need not redistribute them.

DocVQA and InfographicVQA are alternative document sources through the official RRC route. [Creator documentation](https://site.docvqa.org/datasets) specifies login and download terms. Authenticated access was not verified during this source assessment; the completed study used the author-published TAT-DQA assets described above.

For all sources, pretraining exposure is unknown. A selected training panel is a development diagnostic, not an uncontaminated transfer benchmark. V*Bench, HR-Bench, and TextVQA remain reserved.
