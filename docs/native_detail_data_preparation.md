# Reproducing the native-detail development panel

This document records the completed, outcome-blind preparation of **100 main examples and two separate hardware-smoke examples**, one question per original financial report. The sources are the official TAT-DQA training release. This is a development diagnostic, not a held-out benchmark or evidence of uncontaminated transfer. See [source access and scoring provenance](native_detail_data_sources.md) for the creator links and terms.

## Pinned release

The [author's public data folder](https://drive.google.com/drive/folders/1SGpZyRWqycMd_dZim1ygvWhl5KdJYDR2) supplied these assets. The local names differ from the original listing; file identities and hashes are pinned rather than inferred from filenames.

| Original download name | Local name | Drive file ID | Bytes |
|---|---|---|---:|
| `tatdqa_dataset_train.json` | `tat_dqa_dataset_train.json` | `1FQUbZRlJKB-1sbvZ_HiZtE5isNwSvjki` | 9,722,633 |
| `tatdqa_docs_train.zip` | `tat_dqa_train.zip` | `1VX4U0wx4ojITs72d5FAKvNJMkL5J6Xdw` | 1,172,553,879 |

Downloaded on 25 September 2026 at 20:19:35 UTC and 20:20:17 UTC, respectively. SHA-256 values:

- Annotations: `3025b4ca9c3c87f5ecd2c504371e1ee92929ef1f31d8a3ac38bc3a5376582fab`
- Document archive: `412dde804ec9d5888b888a9d7179f2157bc2679b92379f9804e6f2062e7237d9`

The archive contains 2,207 PDFs, 2,207 associated OCR JSON files and 2,448 released PNGs. The released PNGs are **224 × 224 thumbnails** and were not used. `doc.page` denotes starting-page metadata; page count is checked using the OCR `pages` array and the actual PDF page tree.

## Selection and technical screens

The release has 13,251 questions over 2,207 document entries. We retain questions whose OCR document has exactly one page, `answer_type="span"`, one nonempty string answer, empty scale, `req_comparison=false`, and an answer no longer than 160 characters or 25 whitespace-separated words. These metadata filters leave **2,666 questions from 167 original report sources**. Those 167 reports are metadata candidates; we did not claim to inspect every candidate PDF.

The recorded first-failure counts are: 1,429 questions in non-single-page OCR documents, 6,699 outside the span type, 1,400 with nonempty scale, 632 requiring comparison and 425 exceeding the answer-length limit. These are ordered exclusions, not overlapping reason totals.

With seed `20260925`, reports are sorted by SHA-256 of `seed:doc.source`, then their questions by SHA-256 of `seed:question.uid`. For each report, take the first technically eligible question. Actual PDF eligibility requires exactly one page, at least 100 stripped characters of extractable text, and sufficient page area when rendered at 200 DPI. The final RGB raster must have both dimensions at least 32 pixels and area at least `2 × 1024 × 32 × 32 = 2,097,152` pixels. This area rule is a preparation screen, not a claim of equal realized model tokens or compute.

The first 102 reports passed on their first screened PDF; no PDF-screen or rendering exclusions occurred. The first two reports in the deterministic order are smoke examples; the following 100 are main. Thus all 102 original sources are distinct. Selection never uses model outputs, difficulty, answer location, or agreement between the reference and evidence text. Later source-audit flags do not alter this selection or its labels.

## Rendering and local reproduction

Run [the preparation script](../experiments/prepare_tatdqa.py) with Python, Pillow, pypdf and Poppler installed. Place the two pinned downloads under their local names in the raw directory. The output directory must be absent or empty; the script refuses to overwrite an existing panel.

```text
python experiments/prepare_tatdqa.py --raw /path/to/tatdqa_raw --output /path/to/new_panel --pdftoppm /path/to/pdftoppm
```

Only the chosen 102 PDFs are extracted and rendered. ZIP members are explicit, validated document-UID paths; upstream code is not executed. The recorded renderer is **Poppler pdftoppm 26.07.0**, using full MediaBox pages and `-f 1 -singlefile -r 200 -png`. Output is RGB, without resizing the source render or enlarging released thumbnails. The local preparation took **42.05 seconds**, including source-hash checks, metadata screening, extraction and rendering; this is an observed local duration, not a portable performance guarantee.

Actual raster areas range from **2,735,716 to 4,550,454 pixels**. Every generated image was checked again for its SHA-256, RGB mode, dimensions and minimum area; IDs and original report groups were checked for uniqueness. PDF text extraction is only a technical screen: it does not prove that all relevant marks are vector content, that every page is visually faithful, or that a reference is correct.

## Frozen identities and review coverage

Selection was frozen at **2026-09-25 20:34:42 UTC**. The original local archive retains the selection metadata and both manifests; the public [study index](../reports/README.md) summarizes this panel. Historical bindings were:

- Main manifest SHA-256: `757c3a4d024ac726f701c8732424e3457bfb292d224f67e2de22201ac641adc5`
- Smoke manifest SHA-256: `ac4f35d9edf4c6e6a7b70225668fc89172a618f3eaecaa661009269c2c109564`
- Preparation-script SHA-256: `83f223acec84d40b902ab31fa927bb0fd989e5822e5f10dcb095c4e641da9787`

Manifest image paths remain relative to the generated data directory. Archived manifests alone are identity records and cannot load images from the repository; regenerate the data directory before running inference. Source images, PDFs, full OCR content and the source-linked manifests are not included in this numerical release. Renderer identity, per-page hashes, timing, dimensions, source-report IDs and the deterministic selection walk are preserved in metadata or manifests.

All 102 question/reference pairs and their available mapped evidence text received source-only qualitative review. Suspicious or incomplete mappings received additional full-page text review, and every secondary exclusion received a complete rendered-page inspection. These descriptive checks do not constitute independent expert validation or visual certification of all 102 pages. The original source audit records coverage, remaining caveats and reviewed IDs; it is retained locally rather than distributed as source text.

The original local quality mask was locked at **20:41:14 UTC**, before main inference began at 20:41:23 UTC. Primary analysis retains all 100 original examples; a prespecified sensitivity panel retains 94 after six confirmed question/reference or source-page concerns. Mapping-only defects and supported partial answers remain included. No reference was corrected. Mask SHA-256: `a621e6124f866ec4a1d0b882dc03cd5a10fe3bcfd8e567574c65934c76f9db90`.
