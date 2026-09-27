# Run the model pipeline

The [numerical replay](reproduce.md) works with Python alone. Model
inference uses a CUDA-capable NVIDIA GPU, the pinned Qwen3-VL snapshot, and source
images downloaded from the original dataset provider.

## Environment

The measured setup uses Python 3.12, PyTorch 2.8.0, Transformers 4.57.6 and an RTX 5090.
Create a fresh environment, activate it for your shell, and install:

```sh
python -m venv .venv
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` in PowerShell before the installation commands.
Use the PyTorch build appropriate for your GPU.

## GQA smoke run

The following commands download the pinned model and GQA sample, run four images,
and generate an analysis and local gallery:

```sh
python -m lookagain.cli fetch-model --output models/qwen3-vl-4b --revision ebb281ec70b05090aa6165b016eac8ec08e71b17
python -m lookagain.data --output data/gqa_dev400 --limit 400 --revision a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8
python -m lookagain.cli run --manifest data/gqa_dev400/manifest.jsonl --model-dir models/qwen3-vl-4b --config configs/pilot.json --output runs/smoke4 --limit 4
python -m lookagain.cli analyze --run runs/smoke4 --output generated/smoke4
python -m lookagain.cli visualize --run runs/smoke4 --manifest data/gqa_dev400/manifest.jsonl --summary generated/smoke4/summary.json --output local-gallery
```

Inspect the realized image grids, answers, memory and timings. A new output directory
keeps the smoke run separate from a subsequent full panel. Dataset images and local
galleries stay outside the source repository.

## Learned ranking implementation

The learned selector has four stages:

| Stage | Implementation |
| --- | --- |
| Construct controlled requests | `experiments/refinement_core.py` |
| Extract frozen decoder features | `experiments/refinement_features.py` |
| Fit projected ridge rankers | `experiments/refinement_ranker.py`, `experiments/train_refinement.py` |
| Execute and analyze the study | `experiments/refinement_execution.py`, `experiments/analyze_refinement.py` |

Read [the model specification](refinement_model.md) and [fixed study protocol](refinement_protocol.md)
alongside each module's `--help`. The executors validate source manifests,
configurations and prerequisite runs. New experiments require their own source
manifests, configurations, output directories and execution locks.

The [published coefficient sets](../artifacts/ranker_coefficients.json) contain full,
image+position and position-only fits on the 124-source development panel. The
reported ranking result uses five-fold out-of-fold predictions; the coefficients
are the full-data fits for inspecting the learned model.
