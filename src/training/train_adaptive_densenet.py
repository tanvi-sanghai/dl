#!/usr/bin/env python3
"""
Training script for Adaptive DenseNet-121 with enhanced feature selection
Follows the established training patterns in the repository
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..models.densenet121_adaptive import build_densenet121_adaptive
from ..data_pipeline.dataloaders import build_dataloaders
from .engine import train_model
from ..analysis.config import OUTPUT_CONFIG


def run_adaptive_densenet_experiment(
    data_root: str = "dataset",
    output_dir: str = "training_logs/densenet121_adaptive",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.01,
    optimizer: str = "sgd",
    aug_strength: str = "medium",
    label_smoothing: float = 0.1,
    use_weighted_sampler: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict:
    """
    Train Adaptive DenseNet-121 with configurable parameters
    
    Args:
        data_root: Root directory containing train/val/test splits
        output_dir: Directory to save model weights and logs
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        optimizer: Optimizer choice ('sgd' or 'adamw')
        aug_strength: Data augmentation strength ('weak', 'medium', 'strong')
        label_smoothing: Label smoothing factor
        use_weighted_sampler: Whether to use weighted sampling for class imbalance
        num_workers: Number of dataloader workers
        seed: Random seed for reproducibility
        device: Device to train on (None for auto-detect)
    
    Returns:
        Dictionary containing training results and metrics
    """
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build model
    print("[Adaptive DenseNet] Building model with enhanced feature selection...")
    model = build_densenet121_adaptive(num_classes=11)
    
    arch_name = "densenet121_adaptive"
    
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Adaptive DenseNet] Using device: {device}")
    
    # Build dataloaders
    print(f"[Adaptive DenseNet] Loading data from {data_root}")
    dataloaders = build_dataloaders(
        data_root=data_root,
        input_size=(224, 224),
        batch_size=batch_size,
        num_workers=num_workers,
        aug_strength=aug_strength,
        use_weighted_sampler=use_weighted_sampler,
    )
    
    # Training configuration
    run_tag = f"{arch_name}_{optimizer}_ls{label_smoothing}_{aug_strength}_lr{lr}"
    run_dir = os.path.join(output_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    
    # Train model using the engine
    print(f"[Adaptive DenseNet] Starting training for {epochs} epochs...")
    results = train_model(
        model=model,
        dataloaders=dataloaders,
        num_classes=11,
        out_dir=run_dir,
        epochs=epochs,
        optimizer_name=optimizer,
        lr=lr,
        weight_decay=1e-4,
        label_smoothing=label_smoothing,
        run_tag=run_tag,
        scheduler="step",        # use StepLR by default for this script
        warmup_epochs=0,         # ignored for step schedule
        min_lr=1e-6,             # ignored for step schedule
        step_size=10,
        gamma=0.1,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        grad_clip_norm=None,
    )
    
    # Save additional metrics
    metrics_path = Path(run_dir) / f"{run_tag}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'model': arch_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'optimizer': optimizer,
            'aug_strength': aug_strength,
            'label_smoothing': label_smoothing,
            'final_val_accuracy': results.get('best_val_acc', 0),
            'final_val_loss': results.get('best_val_loss', float('inf')),
        }, f, indent=2)
    
    print(f"[Adaptive DenseNet] Training completed. Results saved to {run_dir}")
    print(f"[Adaptive DenseNet] Best validation accuracy: {results.get('best_val_acc', 0):.4f}")
    
    return results


def compare_with_baseline(adaptive_results: Dict, baseline_path: Optional[str] = None):
    """Compare adaptive DenseNet results with baseline DenseNet"""
    
    if baseline_path is None:
        # Try to find baseline results
        baseline_path = OUTPUT_CONFIG.models_root / "densenet121_summary.json"
    
    if Path(baseline_path).exists():
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        print("\n" + "="*50)
        print("Performance Comparison:")
        print("="*50)
        print(f"Baseline DenseNet-121:")
        print(f"  - Final Val Accuracy: {baseline.get('final_val_accuracy', 0):.4f}")
        print(f"  - Epochs: {baseline.get('epochs', 'N/A')}")
        print(f"\nAdaptive DenseNet-121:")
        print(f"  - Final Val Accuracy: {adaptive_results.get('best_val_acc', 0):.4f}")
        print(f"  - Epochs: {adaptive_results.get('epochs', 'N/A')}")
        
        improvement = (adaptive_results.get('best_val_acc', 0) - baseline.get('final_val_accuracy', 0)) * 100
        print(f"\nImprovement: {improvement:+.2f}%")
        print("="*50)
    else:
        print(f"[Info] Baseline results not found at {baseline_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Adaptive DenseNet-121 with enhanced feature selection")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, default="dataset",
                        help="Root directory containing train/val/test splits")
    parser.add_argument("--output-dir", type=str, default="training_logs/densenet121_adaptive",
                        help="Root directory to save model runs (subfolders per run_tag)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"],
                        help="Optimizer to use")
    
    # Augmentation and regularization
    parser.add_argument("--aug-strength", type=str, default="medium", 
                        choices=["weak", "medium", "strong"],
                        help="Data augmentation strength")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--no-weighted-sampler", action="store_true",
                        help="Disable weighted sampling for class imbalance")
    
    # System arguments
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (cuda/mps/cpu)")
    
    # Comparison
    parser.add_argument("--compare", action="store_true",
                        help="Compare with baseline DenseNet after training")
    parser.add_argument("--baseline-path", type=str, default=None,
                        help="Path to baseline results JSON")
    
    args = parser.parse_args()
    
    # Run training
    results = run_adaptive_densenet_experiment(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        aug_strength=args.aug_strength,
        label_smoothing=args.label_smoothing,
        use_weighted_sampler=not args.no_weighted_sampler,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )
    
    # Compare with baseline if requested
    if args.compare:
        compare_with_baseline(results, args.baseline_path)


if __name__ == "__main__":
    main()
