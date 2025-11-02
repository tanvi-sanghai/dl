from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T

from .organamnist_dataset import OrganAMNISTDataset


def _grayscale_normalization(mean: float = 0.5, std: float = 0.5) -> T.Normalize:
    return T.Normalize(mean=[mean], std=[std])


def build_transforms(input_size: Tuple[int, int], aug_strength: str) -> Dict[str, T.Compose]:
    h, w = input_size
    if aug_strength not in {"weak", "medium", "strong"}:
        raise ValueError("aug_strength must be one of: weak, medium, strong")

    common_train = [
        T.Resize((h, w)),
        T.RandomHorizontalFlip(p=0.5),
    ]

    if aug_strength == "weak":
        aug = [T.RandomRotation(degrees=10)]
    elif aug_strength == "medium":
        aug = [
            T.RandomRotation(degrees=15),
            T.RandomResizedCrop((h, w), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        ]
    else:  # strong
        aug = [
            T.RandomRotation(degrees=20),
            T.RandomResizedCrop((h, w), scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
        ]

    train_tfms = T.Compose([
        T.Grayscale(num_output_channels=1),
        *common_train,
        *aug,
        T.ToTensor(),
        _grayscale_normalization(0.5, 0.5),
    ])

    eval_tfms = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((h, w)),
        T.ToTensor(),
        _grayscale_normalization(0.5, 0.5),
    ])

    return {"train": train_tfms, "eval": eval_tfms}


def _make_sampler_for_class_balance(dataset: OrganAMNISTDataset) -> WeightedRandomSampler:
    labels = [lbl for _img, lbl, _p in [dataset[i] for i in range(len(dataset))]]
    counts = Counter(labels)
    num_samples = len(dataset)
    class_weights = {c: num_samples / (len(counts) * cnt) for c, cnt in counts.items()}
    sample_weights = torch.DoubleTensor([class_weights[lbl] for lbl in labels])
    return WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)


def build_dataloaders(
    data_root: str,
    input_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    aug_strength: str = "strong",
    use_weighted_sampler: bool = True,
    manifest_csv_train: Optional[str] = None,
    manifest_csv_val: Optional[str] = None,
) -> Dict[str, DataLoader]:
    transforms = build_transforms(input_size=input_size, aug_strength=aug_strength)

    train_ds = OrganAMNISTDataset(
        root_dir=data_root,
        split="train",
        transform=transforms["train"],
        manifest_csv=manifest_csv_train,
    )
    val_ds = OrganAMNISTDataset(
        root_dir=data_root,
        split="val",
        transform=transforms["eval"],
        manifest_csv=manifest_csv_val,
        class_to_index=getattr(train_ds, "class_to_index", None),
    )

    train_sampler = _make_sampler_for_class_balance(train_ds) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader}


