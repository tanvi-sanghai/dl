from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class SampleRecord:
    path: str
    label: int


class OrganAMNISTDataset(Dataset):
    """Dataset for grayscale OrganAMNIST-like folder or manifest.

    Supports two modes:
      1) Folder mode: root_dir contains subfolders named by class, each with images
      2) Manifest mode: CSV file with columns: path,label (paths can be relative to root or absolute)
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        manifest_csv: Optional[str] = None,
        class_to_index: Optional[dict[str, int]] = None,
    ) -> None:
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.manifest_csv = manifest_csv
        self.class_to_index = class_to_index or {}
        self.samples: List[SampleRecord] = []

        if self.manifest_csv is not None:
            self._load_from_manifest(self.manifest_csv)
        else:
            split_dir = os.path.join(self.root_dir, split)
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
            self._load_from_folders(split_dir)

    def _load_from_manifest(self, csv_path: str) -> None:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
            if "path" in cols and "label" in cols:
                # Generic manifest with absolute/relative paths
                for row in reader:
                    raw_path = row["path"].strip()
                    label = int(row["label"].strip())
                    full_path = raw_path if os.path.isabs(raw_path) else os.path.join(self.root_dir, raw_path)
                    self.samples.append(SampleRecord(path=full_path, label=label))
            elif "file" in cols and "label" in cols:
                # OrganAMNIST style: file names relative to split image directory
                if self.split == "train":
                    base = os.path.join(self.root_dir, "train", "images_train")
                elif self.split == "val":
                    base = os.path.join(self.root_dir, "val", "images_val")
                else:
                    base = os.path.join(self.root_dir, "test", "images")
                for row in reader:
                    fname = row["file"].strip()
                    label = int(row["label"].strip())
                    full_path = os.path.join(base, fname)
                    self.samples.append(SampleRecord(path=full_path, label=label))
            else:
                raise ValueError("Manifest must contain either ('path','label') or ('file','label') columns")

    def _load_from_folders(self, split_dir: str) -> None:
        class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        if not self.class_to_index:
            self.class_to_index = {name: idx for idx, name in enumerate(class_names)}
        for class_name in class_names:
            class_idx = self.class_to_index[class_name]
            class_dir = os.path.join(split_dir, class_name)
            for root, _dirs, files in os.walk(class_dir):
                for fname in files:
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                        self.samples.append(SampleRecord(path=os.path.join(root, fname), label=class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        record = self.samples[index]
        img = Image.open(record.path).convert("L")  # grayscale
        if self.transform is not None:
            img = self.transform(img)
        label = record.label
        return img, label, record.path


