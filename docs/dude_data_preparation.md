# Reproducing the DUDE source preparation

This guide records the original local experiment workflow. Its commands require the historical source manifests, raw records and identity artifacts, which are not bundled in this numerical release. See the [study index](../reports/README.md) and [current CPU replay](reproduce.md) for the published materials.

This workflow prepares a source-audited experiment; it does not load model weights or generate answers. Read the [prospective protocol](dude_replication_protocol.md) before selecting any examples. A technical candidate is not an approved question, and a source-preparation configuration is not an inference lock.

## Inputs and environment

Use Python 3.11 or newer, `requests`, Pillow, `pypdf`, and a local Poppler `pdftoppm` executable. The optional CPU request check additionally uses the project's pinned Transformers stack and locally cached Qwen processor/tokenizer files. It does not need a GPU or model weight files. The run report records the actual environment and tool hashes.

The creator's full archive is 21,240,289,931 bytes. Allow additional disk space for extracted PDFs/OCR and page renders. The extractor retains only allowlisted training PDFs and Azure OCR, and does not execute the author's loader or other archive code.

- [DUDE official repository](https://github.com/duchallenge-team/dude)
- [Public annotation release](https://zenodo.org/records/7763635)
- [Creator's pinned binary/OCR repository](https://huggingface.co/datasets/jordyvl/DUDE_loader/tree/b3662175d3b2482d711f18559b7acc2a5bccc600)
- [Pinned author evaluator](https://github.com/Jordy-VL/DUDEeval/blob/8bd6deac0f9747e344338e3850f02bf76744d888/evaluate_submission.py)

Keep the annotation JSON and lineage CSV from the earlier metadata census. Their exact SHA256 values are enforced by the adapter. Cross-dataset exclusion additionally consumes the historical TAT-DQA manifests retained with the local research artifacts and the original TAT-DQA training annotation/archive. All released PDF fragments from the 102 previously used original reports are compared by exact bytes; available source-name/URL aliases are compared separately. This is a documented overlap screen, not proof of universal source independence.

The source-identity graph covers the metadata-eligible document universe. It does not audit every training PDF or identify arbitrary repackaged versions of earlier reports. Lineage requires a valid HTTP(S) URL in the supplied CSV; a joined row with an unsupported URL syntax also fails that requirement. Exact URL/PDF/full-text grouping is supplemented by declared text-similarity review flags. Low-text scans, changed pagination and shared templates can evade or confuse these screens, so residual source dependence remains a limitation.

Original documents have heterogeneous source licenses. Download provenance and per-source URLs must be retained. Do not assume the annotation license covers every PDF, or commit the downloaded archive, extracted documents, or rendered page images to this repository.

## Source-only commands

Run from the repository root. Substitute local paths for the inputs and Poppler executable. Keep successive run directories separate; the census and render stages refuse to overwrite nonempty output directories.

```text
python experiments/fetch_dude_archive.py --output data/dude_raw

python experiments/extract_dude_sources.py --archive data/dude_raw/DUDE_train-val-test_binaries.tar.gz --census reports/replication_plan/dude_metadata_feasibility.json --output data/dude_extracted

python experiments/prepare_dude_replication.py census --metadata data/dude_metadata/2023-03-23_DUDE_gt_test_PUBLIC.json --lineage data/dude_metadata/licenses_metadata_datalineage.csv --assets data/dude_extracted --tatdqa-raw data/tatdqa_raw --output data/dude_census

python experiments/prepare_dude_replication.py render-audit --census data/dude_census/technical_census.json --assets data/dude_extracted --output data/dude_audit_0000 --pdftoppm /path/to/pdftoppm --start 0 --limit 800

python experiments/check_dude_requests.py --manifest data/dude_audit_0000/candidate_manifest.jsonl --config configs/dude_replication.json --model-dir /path/to/local/qwen3-vl-4b --output data/dude_audit_0000/request_preflight.json
```

The optional `--limit` request-check argument is an explicitly partial engineering check. Omit it for every row. Omit `--model-dir` only when a pixel/chat check without the actual processor is intended; the report records that limitation. Neither form establishes semantic eligibility. An identical native/degraded second view is reported and retained; it is not an automatic exclusion.

## Source review and freeze

Every proposed final question requires source review, including the full page, selected region, original question/reference/variants, row/date/referent and units. Review only unchanged original references. Record the candidate identity, source-audit hash, reviewer, exact inspected artifacts, decision and rationale. A canonical-reference error excludes the whole selected source cluster; do not substitute another question from it. An unsupported variant can be rejected separately.

Continue through the deterministic source order, with the known probe and 10 additional engineering reservations removed from main. Recorded source-only technical or semantic exclusions are allowed under the protocol. Model output, confidence, likely gain and statistical significance are never selection inputs. If the finite pool cannot yield 660 reviewed main sources, report that failure and revise the scientific scope explicitly before collecting any main outcome.

The `finalize` command accepts one or more `--audit-dir` inputs and a JSONL review ledger. It checks all bindings and refuses a smaller main cohort. It is not a shortcut for unperformed review:

```text
python experiments/prepare_dude_replication.py finalize --census data/dude_census/technical_census.json --audit-dir data/dude_audit_0000 --reviews data/dude_source_reviews.jsonl --output data/dude_final
```

Even a successful source freeze does not authorize the older v5 runner. That runner encodes a different cohort, metric and ten actions. A separately checked DUDE runner/analyzer, complete execution bindings and an explicit pre-inference lock are required for the future four-condition experiment.

## Reproducing the metric cross-check

Obtain `evaluate_submission.py` and `LICENSE` from the pinned author evaluator commit linked above, then run:

```text
python experiments/check_dude_metric_parity.py --upstream /path/to/evaluate_submission.py --license /path/to/LICENSE --output reports/dude_metric_parity_local.json
```

The verifier checks both exact file hashes before parsing the source and runs only the three needed scalar functions on five fixed cases and 1,000 seeded synthetic pairs. This verifies agreement with the author's scalar ANLS, including its boundary and normalization behavior. It is not a test of annotation correctness or a model result.

## Reproducing the PDF parser-boundary checks

Use a Python environment containing `pypdf` (the archived check used 6.10.0). From the repository root, run the 13 parser regression tests and the independent 31-case serialization check:

```text
python -m unittest discover -s tests -p test_dude_pdf_parser.py -v
python experiments/check_dude_pdf_boundary.py --output reports/dude_preparation/pdf_boundary_check.json
```

Confirm that the unittest cases actually run: they report a skip when `pypdf` is absent. The standalone checker requires `pypdf` and exits nonzero on any unexpected acceptance, rejection reason, or earlier parser error. It writes JSON and a neighboring Markdown report with the generator, adapter and pypdf source hashes, serialized PDF tokens, and actual library scalar types. These are engineering fixtures, not new scientific examples. They cover legitimate numeric scalars, inherited and indirect page metadata, invalid Boolean/string values and unsupported rotation/scale. The [study index](../reports/README.md) summarizes the completed studies. Running the boundary checker produces a new local report; source annotation and semantic review remain separate requirements.
