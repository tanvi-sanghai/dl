from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
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
    if not weights_path.exists():
        return None
    weights = np.load(weights_path)
    if weights.shape[0] != num_classes:
        return None
    return torch.tensor(weights, dtype=torch.float32)


def _expand_if_needed(t: torch.Tensor) -> torch.Tensor:
    if t.shape[0] == 1:
        return t.expand(3, -1, -1)
    return t


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    num_workers: int = 4
    seed: int = 42


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _prepare_datasets(config: TrainingConfig) -> Tuple[OrganDataset, OrganDataset, List[int]]:
    train_df = load_labels(DATASET_CONFIG.train_labels)
    val_df = load_labels(DATASET_CONFIG.val_labels)

    if config.max_train_samples is not None:
        train_df = train_df.sample(config.max_train_samples, random_state=config.seed).reset_index(drop=True)
    if config.max_val_samples is not None:
        val_df = val_df.sample(config.max_val_samples, random_state=config.seed).reset_index(drop=True)

    class_labels = sorted(train_df["label"].unique())

    mean = [0.4669, 0.4669, 0.4669]
    std = [0.2796, 0.2796, 0.2796]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(_expand_if_needed),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = OrganDataset(train_df, DATASET_CONFIG.train_images, transform)
    val_dataset = OrganDataset(val_df, DATASET_CONFIG.val_images, transform)

    return train_dataset, val_dataset, class_labels


def _train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0
    for inputs, targets in tqdm(loader, desc="Train", leave=False):
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
def _evaluate(model: nn.Module, loader: DataLoader, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for inputs, targets in tqdm(loader, desc="Validate", leave=False):
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
def _collect_predictions(model: nn.Module, loader: DataLoader, device) -> Tuple[np.ndarray, np.ndarray]:
    preds_list = []
    labels_list = []
    for inputs, targets in tqdm(loader, desc="Collect", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        preds_list.append(preds)
        labels_list.append(targets.numpy())
    return np.concatenate(preds_list), np.concatenate(labels_list)


def run_resnet18_baseline(config: TrainingConfig) -> None:
    ensure_output_directories()
    _set_seed(config.seed)

    train_dataset, val_dataset, class_labels = _prepare_datasets(config)
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

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(config.epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = _evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    OUTPUT_CONFIG.models_root.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_CONFIG.models_root / "resnet18_baseline_weights.pth"
    torch.save(model.state_dict(), model_path)

    preds, targets = _collect_predictions(model, val_loader, device)
    cm = confusion_matrix(targets, preds, labels=class_labels)
    np.save(OUTPUT_CONFIG.models_root / "confusion_matrix.npy", cm)

    per_class_acc = {}
    for label in class_labels:
        mask = targets == label
        per_class_acc[label] = float((preds[mask] == label).mean()) if mask.any() else None
    per_class_df = pd.DataFrame({"label": list(per_class_acc.keys()), "accuracy": list(per_class_acc.values())})
    per_class_df.to_json(OUTPUT_CONFIG.models_root / "per_class_accuracy.json", orient="records", indent=2)

    difficult_pairs = []
    for i, actual in enumerate(class_labels):
        for j, predicted in enumerate(class_labels):
            if actual == predicted:
                continue
            difficult_pairs.append({
                "actual": int(actual),
                "predicted": int(predicted),
                "count": int(cm[i, j]),
            })
    difficult_pairs_df = pd.DataFrame(difficult_pairs).sort_values("count", ascending=False)
    difficult_pairs_df.to_csv(OUTPUT_CONFIG.models_root / "difficult_pairs.csv", index=False)

    plt.figure(figsize=(8, 5))
    epochs = np.arange(1, config.epochs + 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.plot(epochs, history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("ResNet18 Baseline Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFIG.models_root / "training_curves_baseline.png")
    plt.close()

    summary = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "final_val_accuracy": history["val_accuracy"][-1],
    }
    (OUTPUT_CONFIG.models_root / "resnet18_baseline_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18 baseline model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    run_resnet18_baseline(config)


if __name__ == "__main__":
    main()