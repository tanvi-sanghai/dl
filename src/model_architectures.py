from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models
import timm


@dataclass(frozen=True)
class ModelRecipe:
    name: str
    input_size: Tuple[int, int]  # H, W
    default_lr: float
    default_weight_decay: float
    default_batch_size: int
    classifier_dropout: float | None = None


def _adapt_first_conv_to_grayscale(module: nn.Module) -> None:
    """Adapt a typical first conv expecting 3 channels to 1 channel, preserving pretrained weights.

    Strategy: average pretrained RGB kernels across channel dimension to initialize the single-channel kernel.
    """
    if not isinstance(module, nn.Conv2d):
        return

    if module.in_channels == 1:
        return  # already grayscale

    if module.in_channels != 3:
        # If model has unusual first layer, fallback to summing and scaling
        with torch.no_grad():
            weight = module.weight
            reduced = weight.mean(dim=1, keepdim=True)
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=1,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
            )
            new_conv.weight.copy_(reduced)
            if module.bias is not None:
                new_conv.bias.copy_(module.bias)
        return

    # Standard case: 3 -> 1 channels
    with torch.no_grad():
        weight = module.weight  # [out, 3, k, k]
        grayscale_weight = weight.mean(dim=1, keepdim=True)  # [out, 1, k, k]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=1,
            bias=(module.bias is not None),
            padding_mode=module.padding_mode,
        )
        new_conv.weight.copy_(grayscale_weight)
        if module.bias is not None:
            new_conv.bias.copy_(module.bias)


def _replace_first_conv(model: nn.Module, attr_path: str = "conv1") -> None:
    """Replace model's first conv to single-channel using averaged pretrained weights.

    attr_path can be a dotted path; e.g., "features.0" for EfficientNet.
    """
    target = model
    parent = None
    last_name = None
    for name in attr_path.split("."):
        parent = target
        last_name = name
        target = getattr(target, name)

    if not isinstance(target, nn.Conv2d):
        # Some models wrap the first conv differently; try to find the first Conv2d
        for mod_name, mod in parent.named_children():
            if isinstance(mod, nn.Conv2d):
                last_name = mod_name
                target = mod
                break

    if not isinstance(target, nn.Conv2d):
        raise RuntimeError("Could not locate a Conv2d module to adapt to grayscale.")

    # Build new conv and swap in
    with torch.no_grad():
        weight = target.weight
        grayscale_weight = weight.mean(dim=1, keepdim=True)
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=target.out_channels,
            kernel_size=target.kernel_size,
            stride=target.stride,
            padding=target.padding,
            dilation=target.dilation,
            groups=1,
            bias=(target.bias is not None),
            padding_mode=target.padding_mode,
        )
        new_conv.weight.copy_(grayscale_weight)
        if target.bias is not None:
            new_conv.bias.copy_(target.bias)

    setattr(parent, last_name, new_conv)


def _replace_classifier(model: nn.Module, in_features: int, num_classes: int, dropout_p: float | None) -> nn.Module:
    layers: list[nn.Module] = []
    if dropout_p is not None and dropout_p > 0:
        layers.append(nn.Dropout(p=dropout_p))
    layers.append(nn.Linear(in_features, num_classes))
    classifier = nn.Sequential(*layers) if len(layers) > 1 else layers[0]

    # Kaiming init for linear to be safe when replacing
    with torch.no_grad():
        if isinstance(classifier, nn.Linear):
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)
        else:
            for m in classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    return classifier


def build_resnet50(num_classes: int = 11, pretrained: bool = True) -> Tuple[nn.Module, ModelRecipe]:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    _replace_first_conv(model, attr_path="conv1")
    in_features = model.fc.in_features
    model.fc = _replace_classifier(model, in_features, num_classes, dropout_p=None)
    recipe = ModelRecipe(
        name="resnet50",
        input_size=(224, 224),
        default_lr=0.05,
        default_weight_decay=1e-4,
        default_batch_size=64,
    )
    return model, recipe


def build_resnet101(num_classes: int = 11, pretrained: bool = True) -> Tuple[nn.Module, ModelRecipe]:
    weights = models.ResNet101_Weights.DEFAULT if pretrained else None
    model = models.resnet101(weights=weights)
    _replace_first_conv(model, attr_path="conv1")
    in_features = model.fc.in_features
    model.fc = _replace_classifier(model, in_features, num_classes, dropout_p=None)
    recipe = ModelRecipe(
        name="resnet101",
        input_size=(224, 224),
        default_lr=0.05,
        default_weight_decay=1e-4,
        default_batch_size=48,
    )
    return model, recipe


def build_efficientnet_b3(num_classes: int = 11, pretrained: bool = True) -> Tuple[nn.Module, ModelRecipe]:
    weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b3(weights=weights)
    # EfficientNet uses first conv at features.0.0
    _replace_first_conv(model, attr_path="features.0.0")
    in_features = model.classifier[-1].in_features  # classifier = [Dropout, Linear]
    model.classifier[-1] = _replace_classifier(model, in_features, num_classes, dropout_p=0.3)
    recipe = ModelRecipe(
        name="efficientnet_b3",
        input_size=(300, 300),
        default_lr=0.02,
        default_weight_decay=1e-5,
        default_batch_size=32,
        classifier_dropout=0.3,
    )
    return model, recipe


def build_densenet121(num_classes: int = 11, pretrained: bool = True) -> Tuple[nn.Module, ModelRecipe]:
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    _replace_first_conv(model, attr_path="features.conv0")
    in_features = model.classifier.in_features
    model.classifier = _replace_classifier(model, in_features, num_classes, dropout_p=None)
    recipe = ModelRecipe(
        name="densenet121",
        input_size=(224, 224),
        default_lr=0.05,
        default_weight_decay=1e-4,
        default_batch_size=64,
    )
    return model, recipe


def build_vit_s16(num_classes: int = 11, pretrained: bool = True) -> Tuple[nn.Module, ModelRecipe]:
    """Vision Transformer Small/16 for 224×224, adapted for grayscale.

    Prefer ImageNet-21k weights if available, otherwise fall back to ImageNet-1k.
    """
    model: nn.Module
    if pretrained:
        # Try 21k variant first; fall back gracefully
        for name in (
            "vit_small_patch16_224_in21k",
            "vit_small_patch16_224.augreg_in21k",
            "vit_small_patch16_224",
        ):
            try:
                model = timm.create_model(name, pretrained=True, in_chans=1, num_classes=num_classes)
                break
            except Exception:
                model = None  # try next
        if model is None:
            model = timm.create_model("vit_small_patch16_224", pretrained=False, in_chans=1, num_classes=num_classes)
    else:
        model = timm.create_model("vit_small_patch16_224", pretrained=False, in_chans=1, num_classes=num_classes)

    recipe = ModelRecipe(
        name="vit_s16",
        input_size=(224, 224),
        default_lr=5e-4,
        default_weight_decay=5e-2,
        default_batch_size=64,
        classifier_dropout=0.0,
    )
    return model, recipe


def build_vit_b16(num_classes: int = 11, pretrained: bool = True) -> Tuple[nn.Module, ModelRecipe]:
    """Vision Transformer Base/16 for 224×224, adapted for grayscale.

    Prefer ImageNet-21k weights if available, otherwise fall back to ImageNet-1k.
    """
    model: nn.Module
    if pretrained:
        for name in (
            "vit_base_patch16_224_in21k",
            "vit_base_patch16_224.augreg_in21k",
            "vit_base_patch16_224.augreg_in21k_ft_in1k",
            "vit_base_patch16_224",
        ):
            try:
                model = timm.create_model(name, pretrained=True, in_chans=1, num_classes=num_classes)
                break
            except Exception:
                model = None
        if model is None:
            model = timm.create_model("vit_base_patch16_224", pretrained=False, in_chans=1, num_classes=num_classes)
    else:
        model = timm.create_model("vit_base_patch16_224", pretrained=False, in_chans=1, num_classes=num_classes)

    recipe = ModelRecipe(
        name="vit_b16",
        input_size=(224, 224),
        default_lr=3e-4,
        default_weight_decay=5e-2,
        default_batch_size=32,
        classifier_dropout=0.0,
    )
    return model, recipe


def build_swin_tiny(num_classes: int = 11, pretrained: bool = True) -> Tuple[nn.Module, ModelRecipe]:
    """Swin Transformer Tiny for 224×224, adapted for grayscale input."""
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        in_chans=1,
        num_classes=num_classes,
    )

    recipe = ModelRecipe(
        name="swin_tiny",
        input_size=(224, 224),
        default_lr=5e-4,
        default_weight_decay=5e-2,
        default_batch_size=64,
        classifier_dropout=0.0,
    )
    return model, recipe


def build_convnext_tiny(num_classes: int = 11, pretrained: bool = True) -> Tuple[nn.Module, ModelRecipe]:
    """ConvNeXt-Tiny (224x224), adapted for grayscale input."""
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)

    # Adapt stem conv (features[0][0]) to 1 channel
    stem = model.features[0][0]
    if isinstance(stem, nn.Conv2d) and stem.in_channels == 3:
        with torch.no_grad():
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
                bias=False,
            )
            new_conv.weight.copy_(stem.weight.mean(dim=1, keepdim=True))
        model.features[0][0] = new_conv

    # Replace classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = _replace_classifier(model, in_features, num_classes, dropout_p=None)

    recipe = ModelRecipe(
        name="convnext_tiny",
        input_size=(224, 224),
        default_lr=0.01,
        default_weight_decay=5e-4,
        default_batch_size=64,
    )
    return model, recipe

def build_all_models(num_classes: int = 11, pretrained: bool = True) -> Dict[str, Tuple[nn.Module, ModelRecipe]]:
    builders = [
        build_convnext_tiny,
        build_resnet50,
        build_resnet101,
        build_efficientnet_b3,
        build_densenet121,
        build_vit_s16,
        build_vit_b16,
        build_swin_tiny,
    ]
    out: Dict[str, Tuple[nn.Module, ModelRecipe]] = {}
    for fn in builders:
        model, recipe = fn(num_classes=num_classes, pretrained=pretrained)
        out[recipe.name] = (model, recipe)
    return out


