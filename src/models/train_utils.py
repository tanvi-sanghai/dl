from __future__ import annotations

import argparse
import json
import os
import ssl
import certifi
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Fix SSL certificate verification for macOS (needed for PyTorch model downloads)
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from ..analysis.config import DATASET_CONFIG, OUTPUT_CONFIG
from ..analysis.utils import ensure_output_directories, load_labels


class OrganDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: Path, transform: transforms.Compose) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = self.images_dir / row.file
        img = plt.imread(image_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        if img.max() <= 1.0:
            img = (img * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        img = self.transform(img)
        label = int(row.label)
        return img, label


def _load_class_weights(num_classes: int) -> Optional[torch.Tensor]:
    weights_path = OUTPUT_CONFIG.models_root / "class_weights.npy"
    # Allow explicit opt-out via CLI/env
    if os.getenv("NO_CLASS_WEIGHTS", "").strip().lower() in ("1", "true", "yes", "y"):  # type: ignore
        return None
    if not weights_path.exists():
        return None
    weights = None
    try:
        weights = np.load(weights_path, allow_pickle=False)
    except Exception:
        # Fall back for files saved with pickling
        try:
            weights = np.load(weights_path, allow_pickle=True)
        except Exception:
            print(f"[train] Warning: failed to load class weights from {weights_path}; proceeding without.")
            return None

    # Coerce to numeric 1D array
    try:
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    except Exception:
        print(f"[train] Warning: class weights at {weights_path} are not numeric; proceeding without.")
        return None
    if weights.shape[0] != num_classes:
        print(
            f"[train] Warning: class weights length {weights.shape[0]} != num_classes {num_classes}; proceeding without."
        )
        return None
    return torch.tensor(weights, dtype=torch.float32)


def _to_channels_first(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3 and t.shape[-1] in (1, 3):
        return t.permute(2, 0, 1)
    if t.ndim == 2:
        return t.unsqueeze(0)
    return t


class EnsureNumChannels:
    def __init__(self, out_channels: int) -> None:
        self.out_channels = out_channels

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        # t is expected to be CHW after ToTensor
        if t.dim() != 3:
            return t.contiguous() if t.is_sparse is False else t
        in_channels = t.shape[0]
        if in_channels == self.out_channels:
            return t.contiguous()
        if self.out_channels == 1:
            # average RGB to grayscale
            return t.mean(dim=0, keepdim=True).contiguous()
        if self.out_channels == 3 and in_channels == 1:
            # replicate grayscale to RGB
            # use repeat to ensure contiguous memory instead of expand (which returns a view)
            return t.repeat(3, 1, 1).contiguous()
        # Fallback: slice or pad if unexpected
        if in_channels > self.out_channels:
            return t[: self.out_channels].contiguous()
        # pad with zeros
        pad_channels = self.out_channels - in_channels
        padding = t.new_zeros(pad_channels, t.shape[1], t.shape[2])
        return torch.cat([t, padding], dim=0).contiguous()


@dataclass
class TrainingConfig:
    model_name: str = "model"
    input_channels: int = 1
    input_size: int = 224
    epochs: int = 30
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    step_size: int = 10
    gamma: float = 0.1
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    num_workers: int = 4
    seed: int = 42
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    force_mps: bool = False

    @staticmethod
    def from_env(defaults: "TrainingConfig") -> "TrainingConfig":
        def _get_int(name: str, fallback: int) -> int:
            v = os.getenv(name)
            return int(v) if v is not None and v != "" else fallback

        def _get_float(name: str, fallback: float) -> float:
            v = os.getenv(name)
            return float(v) if v is not None and v != "" else fallback

        def _get_str(name: str, fallback: str) -> str:
            v = os.getenv(name)
            return v if v is not None and v != "" else fallback

        def _get_bool(name: str, fallback: bool) -> bool:
            v = os.getenv(name)
            if v is None or v == "":
                return fallback
            return v.strip().lower() in ("1", "true", "yes", "y")

        cfg = TrainingConfig(
            model_name=_get_str("MODEL_NAME", defaults.model_name),
            input_channels=_get_int("INPUT_CHANNELS", defaults.input_channels),
            input_size=_get_int("INPUT_SIZE", defaults.input_size),
            epochs=_get_int("EPOCHS", defaults.epochs),
            batch_size=_get_int("BATCH_SIZE", defaults.batch_size),
            lr=_get_float("LEARNING_RATE", defaults.lr),
            momentum=_get_float("MOMENTUM", defaults.momentum),
            weight_decay=_get_float("WEIGHT_DECAY", defaults.weight_decay),
            step_size=_get_int("STEP_SIZE", defaults.step_size),
            gamma=_get_float("GAMMA", defaults.gamma),
            max_train_samples=None,
            max_val_samples=None,
            num_workers=_get_int("NUM_WORKERS", defaults.num_workers),
            seed=_get_int("SEED", defaults.seed),
            mean=defaults.mean,
            std=defaults.std,
            force_mps=_get_bool("FORCE_MPS", defaults.force_mps),
        )
        return cfg


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def prepare_datasets(config: TrainingConfig) -> Tuple[OrganDataset, OrganDataset, List[int]]:
    train_df = load_labels(DATASET_CONFIG.train_labels)
    val_df = load_labels(DATASET_CONFIG.val_labels)

    if config.max_train_samples is not None:
        train_df = train_df.sample(config.max_train_samples, random_state=config.seed).reset_index(drop=True)
    if config.max_val_samples is not None:
        val_df = val_df.sample(config.max_val_samples, random_state=config.seed).reset_index(drop=True)

    class_labels = sorted(train_df["label"].unique())

    if config.mean is None or config.std is None:
        if config.input_channels == 1:
            mean = [0.4669]
            std = [0.2796]
        else:
            mean = [0.4669, 0.4669, 0.4669]
            std = [0.2796, 0.2796, 0.2796]
    else:
        mean = config.mean
        std = config.std

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor(),
        EnsureNumChannels(config.input_channels),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = OrganDataset(train_df, DATASET_CONFIG.train_images, transform)
    val_dataset = OrganDataset(val_df, DATASET_CONFIG.val_images, transform)

    return train_dataset, val_dataset, class_labels


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0
    for inputs, targets in tqdm(loader, desc="Train", unit="batch", leave=True, dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        sample_count += targets.size(0)
    return running_loss / sample_count


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for inputs, targets in tqdm(loader, desc="Validate", unit="batch", leave=True, dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * targets.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device) -> Tuple[np.ndarray, np.ndarray]:
    preds_list = []
    labels_list = []
    for inputs, targets in tqdm(loader, desc="Collect", unit="batch", leave=True, dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        preds_list.append(preds)
        labels_list.append(targets.numpy())
    return np.concatenate(preds_list), np.concatenate(labels_list)


def run_training(build_model_fn: Callable[[int], nn.Module], defaults: TrainingConfig) -> None:
    ensure_output_directories()
    config = TrainingConfig.from_env(defaults)
    _set_seed(config.seed)

    train_dataset, val_dataset, class_labels = prepare_datasets(config)
    num_classes = len(class_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Device selection with ConvNeXt MPS workaround
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    
    # ConvNeXt has known issues with MPS backend (view/stride errors in LayerNorm backward)
    if config.model_name == "convnext_tiny" and use_mps and not config.force_mps:
        print("[train] Warning: ConvNeXt has compatibility issues with MPS backend. Falling back to CPU.")
        print("[train] For faster training, consider using CUDA or a different architecture.")
        print("[train] To force MPS anyway, use --force-mps flag (may crash during training).")
        device = torch.device("cpu")
    else:
        if config.model_name == "convnext_tiny" and use_mps and config.force_mps:
            print("[train] Warning: Forcing MPS for ConvNeXt. This may cause view/stride errors during training.")
        device = torch.device(
            "mps" if use_mps
            else ("cuda" if use_cuda else "cpu")
        )

    model = build_model_fn(num_classes)
    model = model.to(device)

    class_weights = _load_class_weights(num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"[{config.model_name}] Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    OUTPUT_CONFIG.models_root.mkdir(parents=True, exist_ok=True)
    weights_path = OUTPUT_CONFIG.models_root / f"{config.model_name}_weights.pth"
    torch.save(model.state_dict(), weights_path)

    preds, targets = collect_predictions(model, val_loader, device)
    cm = confusion_matrix(targets, preds, labels=class_labels)
    np.save(OUTPUT_CONFIG.models_root / "confusion_matrix.npy", cm)
    # Also save per-model versions to avoid overwrites when running multiple trainings
    try:
        np.save(OUTPUT_CONFIG.models_root / f"confusion_matrix_{config.model_name}.npy", cm)
    except Exception:
        pass

    per_class_acc = {}
    for label in class_labels:
        mask = targets == label
        per_class_acc[label] = float((preds[mask] == label).mean()) if mask.any() else None
    per_class_df = pd.DataFrame({"label": list(per_class_acc.keys()), "accuracy": list(per_class_acc.values())})
    per_class_df.to_json(OUTPUT_CONFIG.models_root / "per_class_accuracy.json", orient="records", indent=2)
    try:
        per_class_df.to_json(
            OUTPUT_CONFIG.models_root / f"per_class_accuracy_{config.model_name}.json",
            orient="records",
            indent=2,
        )
    except Exception:
        pass

    plt.figure(figsize=(8, 5))
    epochs_arr = np.arange(1, config.epochs + 1)
    plt.plot(epochs_arr, history["train_loss"], label="Train Loss")
    plt.plot(epochs_arr, history["val_loss"], label="Val Loss")
    plt.plot(epochs_arr, history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"{config.model_name} Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFIG.models_root / f"training_curves_{config.model_name}.png")
    plt.close()

    summary = {
        "model_name": config.model_name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "final_val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else None,
        "input_size": config.input_size,
        "input_channels": config.input_channels,
    }
    (OUTPUT_CONFIG.models_root / f"{config.model_name}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


def add_common_cli(parser: argparse.ArgumentParser, defaults: TrainingConfig) -> argparse.ArgumentParser:
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--momentum", type=float, default=defaults.momentum)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--step-size", type=int, default=defaults.step_size)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--input-size", type=int, default=defaults.input_size)
    parser.add_argument("--force-mps", action="store_true", help="Force MPS device even for ConvNeXt (may crash)")
    return parser


