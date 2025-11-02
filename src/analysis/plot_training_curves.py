from __future__ import annotations

import json
import os
from typing import Dict

import matplotlib.pyplot as plt


def plot_metrics_json(metrics_json_path: str, out_png_path: str) -> None:
    with open(metrics_json_path, "r") as f:
        hist = json.load(f)
    epochs = [e["epoch"] for e in hist["train"]]
    train_loss = [e["loss"] for e in hist["train"]]
    val_loss = [e["loss"] for e in hist["val"]]
    train_acc = [e["accuracy"] for e in hist["train"]]
    val_acc = [e["accuracy"] for e in hist["val"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close(fig)









