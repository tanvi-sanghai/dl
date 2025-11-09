from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils.metrics import EpochMetrics, aggregate_epoch_metrics
from ..utils.checkpointing import save_checkpoint, ensure_dir
from ..analysis.plot_training_curves import plot_metrics_json


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_criterion(num_classes: int, label_smoothing: float = 0.0) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def _maybe_create_mixup(num_classes: int, mixup_alpha: float, cutmix_alpha: float, label_smoothing: float):
    try:
        from timm.data import Mixup
    except Exception:
        return None
    if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
        return Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.0,
            mode='batch',
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        )
    return None


def _create_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int, min_lr: float = 1e-6):
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: (e + 1) / max(1, warmup_epochs),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    epoch_index: int,
    progress_desc: str = "train",
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    label_smoothing: float = 0.0,
    grad_clip_norm: Optional[float] = None,
) -> EpochMetrics:
    model.train()
    running_losses: list[float] = []
    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    pbar = tqdm(dataloader, desc=f"{progress_desc} | epoch {epoch_index}", leave=False)
    mixup_fn = _maybe_create_mixup(num_classes, mixup_alpha, cutmix_alpha, label_smoothing)
    for images, targets, _paths in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if mixup_fn is not None:
            images, targets_onehot = mixup_fn(images, targets)
            outputs = model(images)
            loss = _soft_cross_entropy(outputs, targets_onehot)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        running_losses.append(float(loss.item()))
        logits_list.append(outputs.detach())
        targets_list.append(targets.detach())
        pbar.set_postfix({"loss": f"{running_losses[-1]:.4f}"})

    return aggregate_epoch_metrics(running_losses, logits_list, targets_list, num_classes)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    epoch_index: int,
    progress_desc: str = "val",
) -> EpochMetrics:
    model.eval()
    losses: list[float] = []
    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    pbar = tqdm(dataloader, desc=f"{progress_desc} | epoch {epoch_index}", leave=False)
    for images, targets, _paths in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        losses.append(float(loss.item()))
        logits_list.append(outputs.detach())
        targets_list.append(targets.detach())
        pbar.set_postfix({"loss": f"{losses[-1]:.4f}"})

    return aggregate_epoch_metrics(losses, logits_list, targets_list, num_classes)


def create_optimizer(optimizer_name: str, model: nn.Module, lr: float, weight_decay: float) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError("optimizer_name must be 'sgd' or 'adamw'")


def save_metrics_json(metrics: Dict, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)


def maybe_update_best(
    best: dict,
    val_metrics: EpochMetrics,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    out_dir: str,
    tag: str,
) -> dict:
    is_better = (best.get("accuracy", -1.0) < val_metrics.accuracy)
    if is_better:
        best = {
            "accuracy": val_metrics.accuracy,
            "loss": val_metrics.loss,
            "epoch": epoch,
        }
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best": best,
            },
            filename=os.path.join(out_dir, f"best_{tag}.pth"),
        )
    return best


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    num_classes: int,
    out_dir: str,
    epochs: int,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    run_tag: str,
    scheduler: str | None = None,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
    step_size: int = 10,
    gamma: float = 0.1,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    grad_clip_norm: Optional[float] = None,
) -> Dict:
    device = resolve_device()
    model.to(device)

    criterion = build_criterion(num_classes=num_classes, label_smoothing=label_smoothing)
    optimizer = create_optimizer(optimizer_name=optimizer_name, model=model, lr=lr, weight_decay=weight_decay)
    lr_scheduler = None
    if scheduler == "cosine":
        lr_scheduler = _create_scheduler(optimizer, epochs=epochs, warmup_epochs=warmup_epochs, min_lr=min_lr)
    elif scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    history = {"train": [], "val": []}
    best = {}

    for epoch in range(1, epochs + 1):
        train_m = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            criterion,
            device,
            num_classes,
            epoch,
            progress_desc="train",
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            label_smoothing=label_smoothing,
            grad_clip_norm=grad_clip_norm,
        )
        val_m = evaluate(model, dataloaders["val"], criterion, device, num_classes, epoch, progress_desc="val")
        if lr_scheduler is not None:
            lr_scheduler.step()

        history["train"].append({"epoch": epoch, "loss": train_m.loss, "accuracy": train_m.accuracy})
        history["val"].append({
            "epoch": epoch,
            "loss": val_m.loss,
            "accuracy": val_m.accuracy,
            "per_class_accuracy": val_m.per_class_accuracy,
        })

        # Save last checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best": best,
            },
            filename=os.path.join(out_dir, f"last_{run_tag}.pth"),
        )

        best = maybe_update_best(best, val_m, model, optimizer, epoch, out_dir, run_tag)

    # Save final metrics json
    metrics_path = os.path.join(out_dir, f"metrics_{run_tag}.json")
    save_metrics_json(history, metrics_path)
    # Save training curves plot
    try:
        plot_metrics_json(metrics_path, os.path.join(out_dir, f"curves_{run_tag}.png"))
    except Exception as e:
        # Non-fatal
        print(f"Plotting failed: {e}")
    return {"history": history, "best": best}


