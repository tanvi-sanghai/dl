"""Model training utilities and builders (lazy imports).

This module exposes builder functions without eagerly importing submodules.
Lazy import avoids runpy warnings when executing submodules via `python -m`.
"""

__all__ = [
    "build_resnet50",
    "build_resnet101",
    "build_efficientnet_b3",
    "build_densenet121",
    "build_convnext_tiny",
    "build_resnext50_32x4d",
    "build_resnext101_32x8d",
]


def build_resnet50(*args, **kwargs):
    from .resnet50 import build_resnet50 as _fn
    return _fn(*args, **kwargs)


def build_resnet101(*args, **kwargs):
    from .resnet101 import build_resnet101 as _fn
    return _fn(*args, **kwargs)


def build_efficientnet_b3(*args, **kwargs):
    from .efficientnet_b3 import build_efficientnet_b3 as _fn
    return _fn(*args, **kwargs)


def build_densenet121(*args, **kwargs):
    from .densenet121 import build_densenet121 as _fn
    return _fn(*args, **kwargs)


def build_convnext_tiny(*args, **kwargs):
    from .convnext_tiny import build_convnext_tiny as _fn
    return _fn(*args, **kwargs)


def build_resnext50_32x4d(*args, **kwargs):
    from .resnext50_32x4d import build_resnext50_32x4d as _fn
    return _fn(*args, **kwargs)


def build_resnext101_32x8d(*args, **kwargs):
    from .resnext101_32x8d import build_resnext101_32x8d as _fn
    return _fn(*args, **kwargs)


