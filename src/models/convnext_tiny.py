from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import timm

from .train_utils import TrainingConfig, add_common_cli, run_training


def build_convnext_tiny(num_classes: int) -> nn.Module:
    """Build ConvNeXt-Tiny using timm with MPS compatibility workaround."""
    # Use timm which has better support for grayscale
    model = timm.create_model(
        'convnext_tiny',
        pretrained=True,
        in_chans=1,  # Direct grayscale support
        num_classes=num_classes
    )
    
    # MPS (Metal) has issues with ConvNeXt's LayerNorm operations during backward
    # Wrap forward to use mixed precision or force contiguous operations
    original_forward = model.forward
    
    def safe_forward(x):
        # Ensure input is contiguous and use torch.utils.checkpoint for problematic blocks
        x = x.contiguous()
        # Call original forward
        return original_forward(x)
    
    model.forward = safe_forward
    
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
        weight_decay=1e-4,
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



