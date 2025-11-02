from __future__ import annotations

import json
import os
import re
from typing import Dict, List

import csv


def parse_tag_from_filename(name: str) -> Dict[str, str]:
    # Expected run_tag: {arch}_{opt}_ls{ls}_{aug}_lr{lr}
    m = re.match(r"(.+?)_(sgd|adamw)_ls([0-9.]+)_(weak|medium|strong)_lr([0-9.]+)", name)
    if not m:
        return {}
    return {
        "architecture": m.group(1),
        "optimizer": m.group(2),
        "label_smoothing": m.group(3),
        "augmentation": m.group(4),
        "lr": m.group(5),
    }


def main(log_root: str = "training_logs", out_csv: str = "analysis_outputs/tables/model_comparison_clean.csv") -> None:
    rows: List[List[str]] = []
    headers = ["architecture", "optimizer", "label_smoothing", "augmentation", "lr", "best_val_accuracy", "best_epoch"]

    for arch_dir in sorted(os.listdir(log_root)):
        arch_path = os.path.join(log_root, arch_dir)
        if not os.path.isdir(arch_path):
            continue
        for run_dir in sorted(os.listdir(arch_path)):
            run_path = os.path.join(arch_path, run_dir)
            metrics_path = os.path.join(run_path, f"metrics_{run_dir}.json")
            if not os.path.isfile(metrics_path):
                continue
            with open(metrics_path, "r") as f:
                hist = json.load(f)
            best_acc = 0.0
            best_epoch = 0
            for e in hist.get("val", []):
                if e["accuracy"] > best_acc:
                    best_acc = e["accuracy"]
                    best_epoch = e["epoch"]
            meta = parse_tag_from_filename(run_dir)
            if not meta:
                continue
            rows.append([
                meta["architecture"], meta["optimizer"], meta["label_smoothing"], meta["augmentation"], meta["lr"],
                f"{best_acc:.4f}", str(best_epoch)
            ])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    main()


