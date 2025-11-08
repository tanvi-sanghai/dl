from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from ..analysis.config import DATASET_CONFIG, OUTPUT_CONFIG
from ..analysis.utils import ensure_output_directories
from .train_utils import TrainingConfig, EnsureNumChannels
from .resnet50 import build_resnet50
from .resnet101 import build_resnet101
from .efficientnet_b3 import build_efficientnet_b3
from .densenet121 import build_densenet121
from .convnext_tiny import build_convnext_tiny
from .resnext50_32x4d import build_resnext50_32x4d
from .resnext101_32x8d import build_resnext101_32x8d
from .densenet121_adaptive import build_densenet121_adaptive


class PredictDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, images_dir: Path, transform: transforms.Compose) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        from PIL import Image

        img = Image.open(self.images_dir / row.file).convert("L")
        img = self.transform(img)
        return int(row["index"]), row.file, img


def _build_transform(input_size: int, input_channels: int) -> transforms.Compose:
    if input_channels == 1:
        mean = [0.4669]
        std = [0.2796]
    else:
        mean = [0.4669, 0.4669, 0.4669]
        std = [0.2796, 0.2796, 0.2796]
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        EnsureNumChannels(input_channels),
        transforms.Normalize(mean=mean, std=std),
    ])


def _models_registry() -> Dict[str, Tuple[Callable[[int], nn.Module], TrainingConfig]]:
    return {
        "resnet50": (
            build_resnet50,
            TrainingConfig(model_name="resnet50", input_channels=1, input_size=224),
        ),
        "resnet101": (
            build_resnet101,
            TrainingConfig(model_name="resnet101", input_channels=1, input_size=224),
        ),
        "resnext50_32x4d": (
            build_resnext50_32x4d,
            TrainingConfig(model_name="resnext50_32x4d", input_channels=1, input_size=224),
        ),
        "resnext101_32x8d": (
            build_resnext101_32x8d,
            TrainingConfig(model_name="resnext101_32x8d", input_channels=1, input_size=224),
        ),
        "efficientnet_b3": (
            build_efficientnet_b3,
            TrainingConfig(model_name="efficientnet_b3", input_channels=1, input_size=300),
        ),
        "densenet121": (
            build_densenet121,
            TrainingConfig(model_name="densenet121", input_channels=1, input_size=224),
        ),
        "densenet121_adaptive": (
            build_densenet121_adaptive,
            TrainingConfig(model_name="densenet121_adaptive", input_channels=1, input_size=224),
        ),
        "convnext_tiny": (
            build_convnext_tiny,
            TrainingConfig(model_name="convnext_tiny", input_channels=1, input_size=224),
        ),
    }


@torch.no_grad()
def _infer_model(model_name: str, build_fn: Callable[[int], nn.Module], cfg: TrainingConfig) -> None:
    ensure_output_directories()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Build and load weights
    num_classes = 11
    model = build_fn(num_classes).to(device)
    weights_path = OUTPUT_CONFIG.models_root / f"{model_name}_weights.pth"
    state_dict = None
    if weights_path.exists():
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location=device)
    else:
        # Fallback: locate latest best checkpoint under training_logs/<model_name>/**/best_*.pth
        from pathlib import Path as _P
        logs_root = _P(__file__).resolve().parents[2] / "training_logs" / model_name
        try:
            candidates = sorted(logs_root.rglob("best_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            candidates = []
        if candidates:
            ckpt_path = candidates[0]
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get("model_state", None)
        else:
            raise FileNotFoundError(f"Weights not found for {model_name}: {weights_path} and no best_*.pth under {logs_root}")

    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)  # may raise if invalid; surfaces clearly
    model.eval()

    # Prepare test dataset
    manifest = pd.read_csv(DATASET_CONFIG.test_manifest)
    if not {"index", "file"}.issubset(manifest.columns):
        raise ValueError(f"Test manifest missing required columns: {DATASET_CONFIG.test_manifest}")
    manifest = manifest.sort_values("index").reset_index(drop=True)

    transform = _build_transform(cfg.input_size, cfg.input_channels)

    dataset = PredictDataset(manifest, DATASET_CONFIG.test_images, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    indices: List[int] = []
    files: List[str] = []
    preds: List[int] = []
    probs: List[np.ndarray] = []

    for batch in tqdm(loader, desc=f"Test [{model_name}]", unit="batch", leave=True, dynamic_ncols=True):
        b_indices, b_files, inputs = batch
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = outputs.argmax(dim=1).cpu().numpy().astype(int)

        indices.extend([int(i) for i in b_indices])
        files.extend(list(b_files))
        preds.extend([int(p) for p in predictions])
        probs.extend(list(probabilities))

    # Save outputs per model
    out_dir = OUTPUT_CONFIG.models_root / "predictions" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Submission
    sub_df = pd.DataFrame({"index": indices, "id": preds}).sort_values("index")
    sub_df.to_csv(out_dir / "submission.csv", index=False)

    # Detailed predictions with probabilities
    prob_cols = {f"p{i}": [float(p[i]) for p in probs] for i in range(11)}
    detail_df = pd.DataFrame({"index": indices, "file": files, "pred": preds, **prob_cols}).sort_values("index")
    detail_df.to_csv(out_dir / "test_predictions.csv", index=False)


def main() -> None:
    registry = _models_registry()
    parser = argparse.ArgumentParser(description="Generate test predictions for OrganAMNIST models")
    parser.add_argument("--model", type=str, choices=list(registry.keys()))
    parser.add_argument("--all", action="store_true", help="Run predictions for all models")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if not args.all and not args.model:
        parser.error("Specify --model or --all")

    if args.all:
        for name, (builder, defaults) in registry.items():
            cfg = defaults
            cfg.num_workers = args.num_workers
            try:
                _infer_model(name, builder, cfg)
            except FileNotFoundError as e:
                print(f"[predict] Skipping {name}: {e}")
                continue
        return

    name = args.model
    builder, defaults = registry[name]
    defaults.num_workers = args.num_workers
    try:
        _infer_model(name, builder, defaults)
    except FileNotFoundError as e:
        raise SystemExit(
            f"{e}. Train the model first, e.g.: 'python -m src.models.train --model {name}'"
        )


if __name__ == "__main__":
    main()


