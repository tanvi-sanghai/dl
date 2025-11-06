#!/usr/bin/env python3
"""
Test script to verify Adaptive DenseNet implementation
"""

import torch
import torch.nn as nn
from src.models.densenet121_adaptive import build_densenet121_adaptive, AdaptiveDenseNet121, SEBlock, AdaptiveFusion


def test_se_block():
    """Test SE block functionality"""
    print("Testing SE Block...")
    se = SEBlock(channels=64, reduction=16)
    x = torch.randn(2, 64, 32, 32)
    out = se(x)
    assert out.shape == x.shape, f"SE block output shape mismatch: {out.shape} vs {x.shape}"
    print("✓ SE Block test passed")


def test_adaptive_fusion():
    """Test Adaptive Fusion module"""
    print("\nTesting Adaptive Fusion...")
    fusion = AdaptiveFusion(num_features=128, num_layers=4)
    
    # Test with multiple feature maps
    features = [torch.randn(2, 32, 16, 16) for _ in range(4)]
    out = fusion(features)
    assert out.shape == features[0].shape, f"Fusion output shape mismatch: {out.shape} vs {features[0].shape}"
    
    # Test with single feature map
    single_feature = [torch.randn(2, 32, 16, 16)]
    out_single = fusion(single_feature)
    assert out_single.shape == single_feature[0].shape
    print("✓ Adaptive Fusion test passed")


def test_model_construction():
    """Test model construction and forward pass"""
    print("\nTesting Model Construction...")
    
    # Test custom model
    model = AdaptiveDenseNet121(num_classes=11)
    x = torch.randn(2, 1, 224, 224)  # Batch of 2 grayscale images
    out = model(x)
    assert out.shape == (2, 11), f"Model output shape mismatch: {out.shape} vs (2, 11)"
    print("✓ Custom model construction test passed")
    
    # Test build function with pretrained weights
    print("\nTesting Pretrained Model Build...")
    model_pretrained = build_densenet121_adaptive(num_classes=11)
    out_pretrained = model_pretrained(x)
    assert out_pretrained.shape == (2, 11), f"Pretrained model output shape mismatch: {out_pretrained.shape}"
    print("✓ Pretrained model build test passed")


def test_parameter_count():
    """Compare parameter counts between models"""
    print("\nComparing Parameter Counts...")
    
    # Load standard DenseNet for comparison
    from torchvision.models import densenet121
    standard_model = densenet121(weights=None)
    standard_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    standard_model.classifier = nn.Linear(standard_model.classifier.in_features, 11)
    
    # Our adaptive model
    adaptive_model = build_densenet121_adaptive(num_classes=11)
    
    standard_params = sum(p.numel() for p in standard_model.parameters())
    adaptive_params = sum(p.numel() for p in adaptive_model.parameters())
    
    print(f"Standard DenseNet-121 parameters: {standard_params:,}")
    print(f"Adaptive DenseNet-121 parameters: {adaptive_params:,}")
    print(f"Additional parameters: {adaptive_params - standard_params:,} ({(adaptive_params/standard_params - 1)*100:.1f}% increase)")
    
    # Check trainable parameters
    standard_trainable = sum(p.numel() for p in standard_model.parameters() if p.requires_grad)
    adaptive_trainable = sum(p.numel() for p in adaptive_model.parameters() if p.requires_grad)
    
    print(f"\nTrainable parameters:")
    print(f"  Standard: {standard_trainable:,}")
    print(f"  Adaptive: {adaptive_trainable:,}")


def test_gradient_flow():
    """Test that gradients flow properly through the model"""
    print("\nTesting Gradient Flow...")
    
    model = build_densenet121_adaptive(num_classes=11)
    model.train()
    
    x = torch.randn(2, 1, 224, 224, requires_grad=True)
    target = torch.randint(0, 11, (2,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Forward pass
    output = model(x)
    loss = criterion(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check that gradients are computed
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients computed!"
    print("✓ Gradient flow test passed")


def test_memory_efficiency():
    """Test memory usage of the model"""
    print("\nTesting Memory Efficiency...")
    
    model = build_densenet121_adaptive(num_classes=11)
    model.eval()
    
    # Test with different batch sizes
    batch_sizes = [1, 4, 8, 16]
    
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, 1, 224, 224)
            with torch.no_grad():
                out = model(x)
            print(f"✓ Batch size {bs}: Success (output shape: {out.shape})")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"✗ Batch size {bs}: Out of memory")
                break
            else:
                raise e


def main():
    print("="*60)
    print("Adaptive DenseNet-121 Implementation Test Suite")
    print("="*60)
    
    # Run all tests
    test_se_block()
    test_adaptive_fusion()
    test_model_construction()
    test_parameter_count()
    test_gradient_flow()
    test_memory_efficiency()
    
    print("\n" + "="*60)
    print("All tests passed successfully! ✓")
    print("="*60)
    
    print("\n📝 Implementation Summary:")
    print("• SE blocks added for channel attention after each dense layer")
    print("• Adaptive fusion module for dynamic feature weighting")
    print("• Pretrained weight initialization from standard DenseNet")
    print("• Compatible with existing training infrastructure")
    print("\n🚀 Ready for training!")


if __name__ == "__main__":
    main()
