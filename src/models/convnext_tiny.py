from __future__ import annotations

import argparse
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from .train_utils import TrainingConfig, add_common_cli, run_training


def build_convnext_tiny(num_classes: int) -> nn.Module:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

    # Adapt stem conv for grayscale (ConvNeXt stem is features[0][0])
    stem_conv = model.features[0][0]
    assert isinstance(stem_conv, nn.Conv2d)
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=stem_conv.out_channels,
        kernel_size=stem_conv.kernel_size,
        stride=stem_conv.stride,
        padding=stem_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        new_conv.weight.copy_(stem_conv.weight.mean(dim=1, keepdim=True))
    model.features[0][0] = new_conv

    # Replace classifier for 11 classes (last layer of classifier)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.classifier[-1].weight, nonlinearity="linear")
    nn.init.zeros_(model.classifier[-1].bias)

    return model


def main() -> None:
    defaults = TrainingConfig(
        model_name="convnext_tiny",
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
    )

    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny on OrganAMNIST")
    add_common_cli(parser, defaults)
    _ = parser.parse_args()

    run_training(build_convnext_tiny, defaults)


if __name__ == "__main__":
    main()



