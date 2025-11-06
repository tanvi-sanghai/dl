OrganAMNIST challenge

Dataset layout
- train/: images and `labels_train.csv`
- val/: images and `labels_val.csv`
- test/: images and `manifest_public.csv` (columns: index,file)

Task
- Train on train and val splits (using their labels)
- Predict on test using `test/manifest_public.csv`
- Submit a CSV with columns `index,pred` for all indices listed in the manifest

Labels
- Classes are integers 0..10

Notes
- Large model weights and arrays are stored via Git LFS; make sure Git LFS is installed before pulling models:
  - Install once: `git lfs install`
  - If you cloned before LFS was enabled: `git lfs fetch --all && git lfs checkout`

Where to look in code
- Training/eval entrypoints live under `src/training/` (e.g., `run_baselines.py`, `run_experiments.py`, `evaluate_best_on_test.py`)
- Model definitions and simple predict utilities are under `src/models/` (e.g., `predict.py`)