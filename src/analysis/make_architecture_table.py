from __future__ import annotations

import csv
from typing import Dict

from ..model_architectures import build_all_models


def main(out_csv: str = "analysis_outputs/tables/architecture_comparison_table.csv") -> None:
    models = build_all_models(num_classes=11, pretrained=True)
    headers = ["architecture", "input_size", "default_lr", "default_weight_decay", "default_batch_size"]
    rows = []
    for name, (_model, recipe) in models.items():
        rows.append([
            recipe.name,
            f"{recipe.input_size[0]}x{recipe.input_size[1]}",
            recipe.default_lr,
            recipe.default_weight_decay,
            recipe.default_batch_size,
        ])

    # Write CSV
    import os
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    main()


