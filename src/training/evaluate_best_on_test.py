from __future__ import annotations

import json
import os
from typing import Dict

import torch
from tqdm.auto import tqdm

from ..data_pipeline.dataloaders import build_transforms
from ..data_pipeline.organamnist_dataset import OrganAMNISTDataset
from ..model_architectures import build_all_models
from .engine import resolve_device


@torch.no_grad()
def evaluate_dir(arch: str, run_tag: str, log_root: str = "training_logs", data_root: str = "IS 2025 OrganAMNIST") -> Dict:
    device = resolve_device()
    models = build_all_models(num_classes=11, pretrained=False)  # we'll load our weights
    model, recipe = models[arch]
    model.to(device)

    ckpt_path = os.path.join(log_root, arch, run_tag, f"best_{run_tag}.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])  # type: ignore
    model.eval()

    tfms = build_transforms(recipe.input_size, aug_strength="weak")["eval"]
    test_ds = OrganAMNISTDataset(root_dir=data_root, split="test", transform=tfms)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=recipe.default_batch_size, shuffle=False, num_workers=4)

    total = 0
    correct = 0
    pbar = tqdm(loader, desc=f"test | {arch}")
    for images, targets, _paths in pbar:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == targets).sum().item())
        total += int(targets.numel())
        pbar.set_postfix({"acc": f"{(correct/total if total else 0):.4f}"})

    acc = correct / total if total else 0.0
    return {"test_accuracy": acc, "num_samples": total}


def main() -> None:
    # Example: evaluate the 8 baselines
    archs = ["resnet50", "resnet101", "efficientnet_b3", "densenet121"]
    opts = ["sgd", "adamw"]
    for arch in archs:
        for opt in opts:
            lr = 0.01 if opt == "sgd" else 0.001
            run_tag = f"{arch}_{opt}_ls0.1_strong_lr{lr}"
            try:
                res = evaluate_dir(arch, run_tag)
                print(arch, opt, res)
            except FileNotFoundError:
                print(f"Skipping missing: {run_tag}")


if __name__ == "__main__":
    main()


