from __future__ import annotations

import argparse
import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

from .train_utils import TrainingConfig, add_common_cli, run_training


def build_densenet121(num_classes: int) -> nn.Module:
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)

    # Adapt first conv for grayscale
    old_conv = model.features.conv0
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
    model.features.conv0 = new_conv

    # Replace classifier for 11 classes
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.classifier.weight, nonlinearity="linear")
    nn.init.zeros_(model.classifier.bias)

    return model


def main() -> None:
    defaults = TrainingConfig(
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
    )

    parser = argparse.ArgumentParser(description="Train DenseNet-121 on OrganAMNIST")
    add_common_cli(parser, defaults)
    _ = parser.parse_args()

    run_training(build_densenet121, defaults)


if __name__ == "__main__":
    main()


