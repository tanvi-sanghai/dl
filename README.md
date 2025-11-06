DL Project – OrganAMNIST

Overview
- This repo trains and evaluates CNN baselines (e.g., ResNet, DenseNet, EfficientNet) on the OrganAMNIST challenge and includes analysis/visualization utilities.
- Large artifacts (model weights `.pth`, arrays `.npy`) are tracked via Git LFS.

Getting started
1) Install system deps
   - Git LFS (required to pull weights): `git lfs install`
2) Python env
   - Create/activate a virtualenv and install deps: `pip install -r requirements.txt`

Data
- Dataset is under `dataset/` with train/val/test splits.
- Test manifest: `test/manifest_public.csv` (columns: `index,file`).

Repo layout
- `dataset/`: dataset splits and manifests
- `src/training/`: training/eval runners (`run_baselines.py`, `run_experiments.py`, `evaluate_best_on_test.py`)
- `src/models/`: model definitions and utilities (`resnet50.py`, `resnet101.py`, `densenet121.py`, `train.py`, `predict.py`)
- `src/analysis/`: analysis scripts to produce figures/tables/reports
- `analysis_outputs/`: generated figures, tables, reports, and trained weights (LFS)

Git LFS
- This repo uses Git LFS for `.pth` and `.npy` files.
- If you cloned before LFS was enabled:
  - `git lfs fetch --all && git lfs checkout`

Typical workflows
- Baselines/experiments: see `src/training/run_baselines.py` and `src/training/run_experiments.py`
- Evaluate best checkpoint on test: `src/training/evaluate_best_on_test.py`
- Quick inference helper: `src/models/predict.py`

Submission (high-level)
- Predict on the test set listed in `test/manifest_public.csv`
- Produce a CSV with columns `index,pred` covering all indices in the manifest

Notes
- If you see placeholders or missing CLI examples, check the scripts’ docstrings for usage and arguments.

