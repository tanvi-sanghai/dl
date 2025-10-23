from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from PIL import Image

from .config import OUTPUT_CONFIG


def ensure_output_directories() -> None:
    OUTPUT_CONFIG.root.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONFIG.analysis_root.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONFIG.models_root.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONFIG.figures.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONFIG.tables.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONFIG.reports.mkdir(parents=True, exist_ok=True)


def load_labels(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Label file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    expected_columns = {"file", "label"}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} missing required columns {expected_columns}")
    return df


def read_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path).convert("L")


def _to_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_to_serializable)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def chunked(iterable: Iterable, size: int) -> Iterable[List]:
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
