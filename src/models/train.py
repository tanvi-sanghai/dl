from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, Tuple

import torch.nn as nn

from .train_utils import TrainingConfig, add_common_cli, run_training
from .resnet50 import build_resnet50
from .resnet101 import build_resnet101
from .efficientnet_b3 import build_efficientnet_b3
from .densenet121 import build_densenet121
from .convnext_tiny import build_convnext_tiny
from .resnext50_32x4d import build_resnext50_32x4d
from .resnext101_32x8d import build_resnext101_32x8d


def _models_registry() -> Dict[str, Tuple[Callable[[int], nn.Module], TrainingConfig]]:
    return {
        "resnet50": (
            build_resnet50,
            TrainingConfig(
                model_name="resnet50",
                input_channels=1,
                input_size=224,
                epochs=30,
                batch_size=64,
                lr=0.01,
                momentum=0.9,
                weight_decay=5e-4,
                step_size=10,
                gamma=0.1,
                num_workers=4,
                seed=42,
            ),
        ),
        "resnet101": (
            build_resnet101,
            TrainingConfig(
                model_name="resnet101",
                input_channels=1,
                input_size=224,
                epochs=30,
                batch_size=32,
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4,
                step_size=10,
                gamma=0.1,
                num_workers=4,
                seed=42,
            ),
        ),
        "efficientnet_b3": (
            build_efficientnet_b3,
            TrainingConfig(
                model_name="efficientnet_b3",
                input_channels=1,
                input_size=300,
                epochs=35,
                batch_size=32,
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-5,
                step_size=12,
                gamma=0.2,
                num_workers=4,
                seed=42,
            ),
        ),
        "densenet121": (
            build_densenet121,
            TrainingConfig(
                model_name="densenet121",
                input_channels=1,
                input_size=224,
                epochs=30,
                batch_size=64,
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4,
                step_size=10,
                gamma=0.1,
                num_workers=4,
                seed=42,
            ),
        ),
        "convnext_tiny": (
            build_convnext_tiny,
            TrainingConfig(
                model_name="convnext_tiny",
                input_channels=1,
                input_size=224,
                epochs=30,
                batch_size=64,
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4,
                step_size=10,
                gamma=0.1,
                num_workers=4,
                seed=42,
            ),
        ),
        "resnext50_32x4d": (
            build_resnext50_32x4d,
            TrainingConfig(
                model_name="resnext50_32x4d",
                input_channels=1,
                input_size=224,
                epochs=30,
                batch_size=64,
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4,
                step_size=10,
                gamma=0.1,
                num_workers=4,
                seed=42,
            ),
        ),
        "resnext101_32x8d": (
            build_resnext101_32x8d,
            TrainingConfig(
                model_name="resnext101_32x8d",
                input_channels=1,
                input_size=224,
                epochs=30,
                batch_size=32,
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4,
                step_size=10,
                gamma=0.1,
                num_workers=4,
                seed=42,
            ),
        ),
    }


def _apply_cli_to_env(args: argparse.Namespace) -> None:
    if args.epochs is not None:
        os.environ["EPOCHS"] = str(args.epochs)
    if args.batch_size is not None:
        os.environ["BATCH_SIZE"] = str(args.batch_size)
    if args.lr is not None:
        os.environ["LEARNING_RATE"] = str(args.lr)
    if args.momentum is not None:
        os.environ["MOMENTUM"] = str(args.momentum)
    if args.weight_decay is not None:
        os.environ["WEIGHT_DECAY"] = str(args.weight_decay)
    if args.step_size is not None:
        os.environ["STEP_SIZE"] = str(args.step_size)
    if args.gamma is not None:
        os.environ["GAMMA"] = str(args.gamma)
    if args.num_workers is not None:
        os.environ["NUM_WORKERS"] = str(args.num_workers)
    if args.seed is not None:
        os.environ["SEED"] = str(args.seed)
    if args.input_size is not None:
        os.environ["INPUT_SIZE"] = str(args.input_size)
    # Optional flag to disable class weights
    if hasattr(args, "no_class_weights") and args.no_class_weights:
        os.environ["NO_CLASS_WEIGHTS"] = "1"
    # Optional flag to force MPS for ConvNeXt
    if hasattr(args, "force_mps") and args.force_mps:
        os.environ["FORCE_MPS"] = "1"


def main() -> None:
    registry = _models_registry()

    default_model = os.getenv("MODEL_NAME", "resnet50")
    parser = argparse.ArgumentParser(description="Unified trainer for OrganAMNIST models")
    parser.add_argument("--model", type=str, default=default_model, choices=list(registry.keys()))
    parser.add_argument("--all", action="store_true", help="Train all models sequentially")
    parser.add_argument("--no-class-weights", action="store_true", help="Ignore class_weights.npy even if present")

    # Attach common hyperparameters using defaults of the selected model
    defaults = registry.get(default_model, list(registry.values())[0])[1]
    add_common_cli(parser, defaults)
    args = parser.parse_args()

    _apply_cli_to_env(args)

    if args.all:
        for name, (builder, defaults) in registry.items():
            os.environ["MODEL_NAME"] = name
            run_training(builder, defaults)
        return

    # Single model
    name = args.model
    builder, defaults = registry[name]
    os.environ["MODEL_NAME"] = name
    run_training(builder, defaults)


if __name__ == "__main__":
    main()


