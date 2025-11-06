from __future__ import annotations

import argparse
import torch
import torch.nn as nn
from torchvision import models

from .train_utils import TrainingConfig, add_common_cli, run_training


def build_resnext101_32x8d(num_classes: int) -> nn.Module:
    model = models.resnext101_32x8d(weights="DEFAULT")

    # Adapt first conv for grayscale
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    model.conv1 = new_conv

    # Replace classifier for dataset num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.fc.weight, nonlinearity="linear")
    if model.fc.bias is not None:
        nn.init.zeros_(model.fc.bias)

    return model


def main() -> None:
    defaults = TrainingConfig(
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
    )

    parser = argparse.ArgumentParser(description="Train ResNeXt-101 (32x8d) on OrganAMNIST")
    add_common_cli(parser, defaults)
    _ = parser.parse_args()

    run_training(build_resnext101_32x8d, defaults)


if __name__ == "__main__":
    main()


