from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
import argparse
from typing import Dict, Iterable, Tuple
import os

from ..model_architectures import build_all_models
from ..data_pipeline.dataloaders import build_dataloaders
from .engine import train_model


@dataclass
class ExperimentConfig:
    architecture: str
    optimizer_name: str  # sgd or adamw
    label_smoothing: float
    aug_strength: str  # weak/medium/strong
    lr: float
    epochs: int
    data_root: str
    out_root: str
    num_workers: int


def run_single(cfg: ExperimentConfig) -> Dict:
    models = build_all_models(num_classes=11, pretrained=True)
    model, recipe = models[cfg.architecture]
    loaders = build_dataloaders(
        data_root=cfg.data_root,
        input_size=recipe.input_size,
        batch_size=recipe.default_batch_size,
        num_workers=cfg.num_workers,
        aug_strength=cfg.aug_strength,
        use_weighted_sampler=True,
    )

    run_tag = f"{cfg.architecture}_{cfg.optimizer_name}_ls{cfg.label_smoothing}_{cfg.aug_strength}_lr{cfg.lr}"
    out_dir = os.path.join(cfg.out_root, cfg.architecture, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    is_transformer = cfg.architecture in {"vit_s16", "vit_b16", "swin_tiny"}
    result = train_model(
        model=model,
        dataloaders=loaders,
        num_classes=11,
        out_dir=out_dir,
        epochs=cfg.epochs,
        optimizer_name=("adamw" if is_transformer else cfg.optimizer_name),
        lr=cfg.lr,
        weight_decay=recipe.default_weight_decay,
        label_smoothing=cfg.label_smoothing,
        run_tag=run_tag,
        scheduler=("cosine" if is_transformer else None),
        warmup_epochs=(5 if is_transformer else 0),
        min_lr=1e-6,
        mixup_alpha=(0.8 if is_transformer else 0.0),
        cutmix_alpha=(1.0 if is_transformer else 0.0),
        grad_clip_norm=(1.0 if is_transformer else None),
    )
    return result


def main() -> None:
    data_root = "IS 2025 OrganAMNIST"  # expects train/ val/ test/ inside
    out_root = "training_logs"
    os.makedirs(out_root, exist_ok=True)

    default_architectures = ["resnet50", "resnet101", "efficientnet_b3", "densenet121", "vit_s16", "vit_b16", "swin_tiny"]
    parser = argparse.ArgumentParser(description="Run experiment sweep")
    parser.add_argument("--architectures", nargs="*", default=None, help="Subset of architectures to run (space-separated)")
    args = parser.parse_args()

    architectures = args.architectures if args.architectures else default_architectures
    optimizers = ["sgd", "adamw"]
    label_smoothings = [0.0, 0.1]
    aug_strengths = ["weak", "medium", "strong"]
    lrs = [1e-2, 5e-3, 1e-3]
    epochs = 50
    # Allow overriding workers through env to avoid fork/shm issues
    num_workers = int(os.getenv("DL_NUM_WORKERS", "4"))

    transformer_set = {"vit_s16", "vit_b16", "swin_tiny"}
    for arch, opt, ls, aug, lr in itertools.product(architectures, optimizers, label_smoothings, aug_strengths, lrs):
        if arch in transformer_set and opt != "adamw":
            continue  # only run AdamW for transformers
        cfg = ExperimentConfig(
            architecture=arch,
            optimizer_name=opt,
            label_smoothing=ls,
            aug_strength=aug,
            lr=lr,
            epochs=epochs,
            data_root=data_root,
            out_root=out_root,
            num_workers=num_workers,
        )
        print(f"Starting: {cfg}")
        run_single(cfg)


if __name__ == "__main__":
    main()


