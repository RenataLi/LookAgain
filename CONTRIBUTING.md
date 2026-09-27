# Contributing to LookAgain

Contributions to evaluation, reproducibility and visual re-observation are welcome.

## Get started

1. Run `python scripts/reproduce_results.py` to verify the released result tables.
2. Read [the architecture](docs/refinement_model.md) and [experiment index](reports/README.md).
3. Open an issue describing the question or implementation change before a substantial experiment.

For CPU development, install Python 3.11+, pytest, Pillow, NumPy, PyTorch,
Matplotlib and pypdf. Run `python -m pytest -q` from the repository root.
The CI workflow installs the CPU build of PyTorch explicitly.

## Experiment changes

Keep archived results and source identities intact. Write new configurations and
outputs to new paths, identify the source panel and its evaluation role, and include
all attempted examples in reported denominators. Report visual and text processing
costs for every policy component used at inference.

For changes to scoring, region geometry or prompts, include a focused test and
describe the expected behavior. For new experiments, provide reproduction commands
and identify the baseline used for comparison.

## Pull requests

Describe the change and the checks you ran. Include
before/after figures for visualization changes. Keep model weights, original
dataset media, credentials and local caches outside the repository.
