from __future__ import annotations

import os
from typing import Any, Dict

import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: Dict[str, Any], filename: str) -> None:
    ensure_dir(os.path.dirname(filename))
    torch.save(state, filename)


def load_checkpoint(filename: str, map_location: str | torch.device | None = None) -> Dict[str, Any]:
    return torch.load(filename, map_location=map_location)


