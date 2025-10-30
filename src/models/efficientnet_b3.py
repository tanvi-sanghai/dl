from __future__ import annotations

import argparse
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from .train_utils import TrainingConfig, add_common_cli, run_training


def build_efficientnet_b3(num_classes: int) -> nn.Module:
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

    # Adapt first conv (stem) for grayscale
    old_conv = model.features[0][0]
    assert isinstance(old_conv, nn.Conv2d)
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
    model.features[0][0] = new_conv

    # Replace classifier for 11 classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.classifier[-1].weight, nonlinearity="linear")
    nn.init.zeros_(model.classifier[-1].bias)

    return model


def main() -> None:
    defaults = TrainingConfig(
        model_name="efficientnet_b3",
        input_channels=1,
        input_size=300,  # EfficientNet-B3 default resolution
        epochs=35,
        batch_size=32,
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-5,
        step_size=12,
        gamma=0.2,
        num_workers=4,
        seed=42,
    )

    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 on OrganAMNIST")
    add_common_cli(parser, defaults)
    _ = parser.parse_args()

    run_training(build_efficientnet_b3, defaults)


if __name__ == "__main__":
    main()


