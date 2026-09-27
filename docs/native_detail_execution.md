# Reproduce the v3 document-detail pilot

This guide records the original local experiment workflow. Its commands require the historical source manifests, raw records and identity artifacts, which are not bundled in this numerical release. See the [study index](../reports/README.md) and [current CPU replay](reproduce.md) for the published materials.

All paths below are relative examples. Keep raw documents, model weights and image-containing galleries outside the source repository. The original workflow produces raw answer records and provenance locally. The public numerical release supplies aggregate-score replay rather than these source-linked records. The protocol and configuration distinguish a 100-report main panel from two separately reserved smoke reports. No controller is trained in this stage.

## Obtain the pinned training data

Follow the [TAT-DQA author site](https://nextplusplus.github.io/TAT-DQA/) and its public data folder. The local run used the author-linked training files `tatdqa_dataset_train.json` and `tatdqa_docs_train.zip`, stored locally as `tat_dqa_dataset_train.json` and `tat_dqa_train.zip`. Only training assets are used; development/test benchmark outcomes were not downloaded for this study. The preparation script verifies SHA256 before reading the archive.

Read [data provenance and licensing notes](native_detail_data_sources.md). The current site/repository and the original paper use different license wording; do not infer redistribution rights for underlying reports from this local research experiment. Images remain outside the source release.

## Prepare original PDF page renders

Use Python with Pillow and pypdf, plus Poppler `pdftoppm` on PATH or an explicit executable path. The local preprocessor used Poppler 26.07.0; exact renderer executable hash and commands were recorded in the original local selection metadata. Do not use the released 224x224 thumbnails. The output directory must be absent/empty to avoid overwriting a frozen panel.

```powershell
python experiments/prepare_tatdqa.py --raw data/tatdqa_raw --output data/tatdqa_v3 --pdftoppm C:/tools/poppler/bin/pdftoppm.exe
```

The script checks one-page PDF/OCR units, renders selected full pages at fixed 200 DPI, and creates `main_manifest.jsonl`, `smoke_manifest.jsonl`, `selection_metadata.json`, and a separate annotation audit pack. It uses no model responses or confidence. See `--help` for cohort parameters, which must match the locked published config when reproducing this run. Preprocessing time is separate from inference time.

The preparation manifests also record measured rendering times, so regenerating them necessarily changes their byte hashes even if the selected pages and image pixels are identical. The **historical strict run** additionally requires the retained original manifest bytes. They are not distributed here. If those local artifacts are available, retain regenerated metadata separately and place the original manifests beside the corresponding images:

```powershell
Copy-Item reports/native100/provenance/main_manifest.jsonl data/tatdqa_v3/main_manifest.jsonl
Copy-Item reports/native100/provenance/smoke_manifest.jsonl data/tatdqa_v3/smoke_manifest.jsonl
```

The runner/analyzer verifies every image against those archived checksums before proceeding. The recorded Windows/Poppler environment may be needed for byte-identical PNGs; a different renderer build can produce different bytes. Treat such a change as a new run identity rather than silently weakening checks. Keep `.gitattributes`: it preserves archived JSON/JSONL bytes and the measured line endings of the pinned metric port, whose bytes participate in the inference digest.

## Check inputs and run locally

Use the project's pinned inference environment (Python 3.12, PyTorch 2.8.0+cu128, Transformers 4.57.6 in the measured run) and locally downloaded frozen model revision. The original `src/lookagain` package is unchanged from v1/v2. Tests need pytest; document preparation additionally needs pypdf and Poppler. The secondary metric uses the existing NumPy dependency and does not need SciPy.

```powershell
python -m pytest -q -p no:cacheprovider
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python experiments/native_detail.py --manifest data/tatdqa_v3/smoke_manifest.jsonl --model-dir models/qwen3-vl-4b --config configs/native_detail.json --output runs/native-smoke2 --smoke
python experiments/native_detail.py --manifest data/tatdqa_v3/main_manifest.jsonl --model-dir models/qwen3-vl-4b --config configs/native_detail.json --output runs/native100
```

The archived smoke run used the initial v3 configuration. Before the main run, a source-only quality-mask sensitivity amendment was added; the main inference settings, source selection, prompts, token ceilings and metrics did not change. Each run embeds its complete actual configuration and inference-source digest. A fresh smoke using the final config produces a different fingerprint from the archived smoke, as expected. Never alter the configuration to resume an existing run.

The runner fails on source-byte mismatch, manifest/count mismatch, changed v1 source code, unexpected processor grids, mismatched paired inputs, or inference errors. It flushes after each action and synchronizes after each report. Output includes `run.json`, all `records.jsonl`, and `completed.json`. A strict analyzer reconstructs the chat and images from the pinned source renders and rescoring rules before reporting any quality result.

## Analyze and inspect

```powershell
python experiments/analyze_native_detail.py --run runs/native100 --manifest data/tatdqa_v3/main_manifest.jsonl --output reports/native100 --quality-mask reports/native100/source_quality_mask.json
python experiments/render_native_detail.py --run reports/native100 --manifest data/tatdqa_v3/main_manifest.jsonl --figure reports/native100/native_detail_results.png --gallery ../lookagain-native-detail-demo/gallery.html
```

The renderer requires the completed run metadata/records and summary together in the run directory. Copy the three completed raw files with the generated summary if analyzing into a separate report directory. Exact analysis CLI and output files are also listed by `--help`.

The local gallery contains up to three sorted examples per outcome stratum and displays all eleven branches, including failures. It is post-hoc illustration, not a representative test set or learned localization demo. Its embedded PNGs reproduce exact model-input pixels; browser display scaling can make details look smaller. The source-code ZIP intentionally excludes that gallery and underlying documents.

## Post-run processor check

The CPU-only supplementary audit reconstructs the actual processor tensors for the first three reports and all four regions. It loads no model weights and performs no generation. It is a post-run engineering check, not a new inferential result.

```powershell
python experiments/audit_native_processor_inputs.py --run reports/native100 --manifest data/tatdqa_v3/main_manifest.jsonl --model-dir models/qwen3-vl-4b --output reports/processor-recomputed
```

The report can be rebuilt with `python experiments/build_native_detail_note.py --summary reports/native100/summary.json --output local-gallery/native-detail-note.pdf`; its interpretation must match the run fingerprint. The three-page PDF was rendered and visually checked. The local HTML gallery passed structural and embedded-image checks.
