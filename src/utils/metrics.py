from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    per_class_accuracy: Dict[int, float]


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def compute_per_class_accuracy(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[int, float]:
    preds = torch.argmax(logits, dim=1)
    per_class_correct = torch.zeros(num_classes, dtype=torch.long, device=targets.device)
    per_class_count = torch.zeros(num_classes, dtype=torch.long, device=targets.device)
    for c in range(num_classes):
        mask = targets == c
        per_class_count[c] = mask.sum()
        per_class_correct[c] = (preds[mask] == c).sum()
    acc = {}
    for c in range(num_classes):
        denom = int(per_class_count[c].item())
        acc[c] = float(per_class_correct[c].item()) / denom if denom > 0 else 0.0
    return acc


def aggregate_epoch_metrics(
    losses: List[float],
    logits_list: List[torch.Tensor],
    targets_list: List[torch.Tensor],
    num_classes: int,
) -> EpochMetrics:
    epoch_loss = float(np.mean(losses)) if losses else 0.0
    logits = torch.cat(logits_list, dim=0) if logits_list else torch.empty(0)
    targets = torch.cat(targets_list, dim=0) if targets_list else torch.empty(0, dtype=torch.long)
    if logits.numel() == 0:
        return EpochMetrics(loss=epoch_loss, accuracy=0.0, per_class_accuracy={})
    acc = compute_accuracy(logits, targets)
    per_class_acc = compute_per_class_accuracy(logits, targets, num_classes)
    return EpochMetrics(loss=epoch_loss, accuracy=acc, per_class_accuracy=per_class_acc)


