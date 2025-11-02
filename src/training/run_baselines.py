from __future__ import annotations

import os
from typing import Dict

import os
from .run_experiments import ExperimentConfig, run_single


def main() -> None:
    # 8 core models: 4 architectures × 2 optimizers, with label smoothing 0.1, strong aug, default LR per grid
    data_root = "IS 2025 OrganAMNIST"
    out_root = "training_logs"
    os.makedirs(out_root, exist_ok=True)

    configs = []
    for arch in ["resnet50", "resnet101", "efficientnet_b3", "densenet121"]:
        for opt in ["sgd", "adamw"]:
            # Use a safe default LR; you can override manually later
            lr = 0.01 if opt == "sgd" else 0.001
            configs.append(ExperimentConfig(
                architecture=arch,
                optimizer_name=opt,
                label_smoothing=0.1,
                aug_strength="strong",
                lr=lr,
                epochs=50,
                data_root=data_root,
                out_root=out_root,
                num_workers=4,
            ))

    # Allow overriding workers through env to avoid fork/shm issues
    num_workers_env = int(os.getenv("DL_NUM_WORKERS", "4"))
    for cfg in configs:
        print(f"Running baseline: {cfg}")
        cfg.num_workers = num_workers_env
        run_single(cfg)


if __name__ == "__main__":
    main()


