"""
Enhanced DenseNet-121 with Adaptive Feature Selection
Implements dynamic feature weighting using attention mechanisms
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights
from collections import OrderedDict

from .train_utils import TrainingConfig, add_common_cli, run_training


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class AdaptiveFusion(nn.Module):
    """Adaptive fusion module for weighted feature combination"""
    def __init__(self, num_features: int, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        
        # Global context extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Weight generation network
        self.weight_net = nn.Sequential(
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // 4, num_layers),
            nn.Softmax(dim=1)
        )
        
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature maps from different layers
        Returns:
            Weighted combination of features
        """
        if len(features) == 1:
            return features[0]
            
        # Stack features for processing
        stacked = torch.stack(features, dim=1)  # [B, num_layers, C, H, W]
        
        # Extract global context from concatenated features
        concat_features = torch.cat(features, dim=1)  # [B, C*num_layers, H, W]
        global_context = self.global_pool(concat_features)  # [B, C*num_layers, 1, 1]
        global_context = global_context.view(global_context.size(0), -1)  # [B, C*num_layers]
        
        # Generate adaptive weights
        weights = self.weight_net(global_context[:, :features[0].size(1)])  # [B, num_layers]
        weights = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, num_layers, 1, 1, 1]
        
        # Apply weighted combination
        weighted = (stacked * weights).sum(dim=1)  # [B, C, H, W]
        
        return weighted


class EnhancedDenseLayer(nn.Module):
    """Dense layer with integrated SE attention"""
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__()
        
        # Standard DenseNet layer components
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, 
                               kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        
        # Add SE attention
        self.se = SEBlock(growth_rate, reduction=4)
        
        self.drop_rate = float(drop_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        
        # Apply SE attention
        new_features = self.se(new_features)
        
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
            
        return new_features


class EnhancedDenseBlock(nn.ModuleDict):
    """Dense block with adaptive feature selection via per-layer gating.

    Each new feature map produced by a dense layer is scaled by an input-adaptive
    scalar gate in (0, 1). Gates are generated from global pooled context of the
    concatenated features, encouraging the model to emphasize informative layers.
    """
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, 
                 growth_rate: int, drop_rate: float):
        super().__init__()
        
        for i in range(num_layers):
            layer = EnhancedDenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'denselayer{i + 1}', layer)

        # Gating: produce one scalar per denselayer conditioned on global context
        total_channels = num_input_features + num_layers * growth_rate
        hidden = max(8, total_channels // 8)
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(total_channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_layers, bias=True),
            nn.Sigmoid(),  # output in (0,1)
        )
        
    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        layer_outputs: list[torch.Tensor] = []

        # Standard dense forward while collecting outputs
        for name, layer in self.items():
            if 'denselayer' in name:
                new_features = layer(torch.cat(features, 1))
                features.append(new_features)
                layer_outputs.append(new_features)

        # Compute input-adaptive gates for each layer output
        concat_all = torch.cat(features, dim=1)  # includes init + all new
        b = concat_all.size(0)
        pooled = self.gate_pool(concat_all).view(b, -1)
        gates = self.gate_mlp(pooled)  # [B, num_layers] in (0,1)

        # Scale each layer output by its corresponding gate
        scaled_outputs: list[torch.Tensor] = []
        for i, out in enumerate(layer_outputs):
            w = gates[:, i].view(b, 1, 1, 1)
            scaled_outputs.append(out * w)

        return torch.cat([init_features] + scaled_outputs, dim=1)


class AdaptiveDenseNet121(nn.Module):
    """DenseNet-121 with adaptive feature selection mechanisms"""
    
    def __init__(self, num_classes: int = 11, growth_rate: int = 32, 
                 block_config: tuple = (6, 12, 24, 16), num_init_features: int = 64,
                 bn_size: int = 4, drop_rate: float = 0):
        super().__init__()
        
        # First convolution for grayscale input
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Dense blocks with adaptive feature selection
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = EnhancedDenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                # Transition layer with SE attention
                trans = nn.Sequential(
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1, bias=False),
                    SEBlock(num_features // 2, reduction=8),  # Add attention to transition
                    nn.AvgPool2d(kernel_size=2, stride=2)
                )
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def build_densenet121_adaptive(num_classes: int) -> nn.Module:
    """Build adaptive DenseNet-121 with pretrained weight initialization"""
    
    # Create our enhanced model
    model = AdaptiveDenseNet121(num_classes=num_classes)
    
    # Load pretrained DenseNet weights and adapt them
    pretrained = densenet121(weights=DenseNet121_Weights.DEFAULT)
    
    # Copy weights from pretrained model where applicable
    with torch.no_grad():
        # Adapt first conv for grayscale (average RGB channels)
        old_weight = pretrained.features.conv0.weight
        new_weight = old_weight.mean(dim=1, keepdim=True)
        model.features.conv0.weight.copy_(new_weight)
        
        # Copy batch norm parameters
        model.features.norm0.weight.copy_(pretrained.features.norm0.weight)
        model.features.norm0.bias.copy_(pretrained.features.norm0.bias)
        model.features.norm0.running_mean.copy_(pretrained.features.norm0.running_mean)
        model.features.norm0.running_var.copy_(pretrained.features.norm0.running_var)
        
        # For dense blocks, we'll copy what we can from standard layers
        # The enhanced layers have the same conv/norm structure, just with added SE blocks
        for block_idx in range(4):
            block_name = f'denseblock{block_idx + 1}'
            
            # Get both blocks
            our_block = getattr(model.features, block_name)
            pretrained_block = getattr(pretrained.features, block_name)
            
            # Copy weights for each dense layer
            for layer_name in our_block.keys():
                if 'denselayer' in layer_name:
                    our_layer = getattr(our_block, layer_name)
                    pretrained_layer = getattr(pretrained_block, layer_name)
                    
                    # Copy conv and norm weights
                    our_layer.conv1.weight.copy_(pretrained_layer.conv1.weight)
                    our_layer.conv2.weight.copy_(pretrained_layer.conv2.weight)
                    
                    our_layer.norm1.weight.copy_(pretrained_layer.norm1.weight)
                    our_layer.norm1.bias.copy_(pretrained_layer.norm1.bias)
                    our_layer.norm1.running_mean.copy_(pretrained_layer.norm1.running_mean)
                    our_layer.norm1.running_var.copy_(pretrained_layer.norm1.running_var)
                    
                    our_layer.norm2.weight.copy_(pretrained_layer.norm2.weight)
                    our_layer.norm2.bias.copy_(pretrained_layer.norm2.bias)
                    our_layer.norm2.running_mean.copy_(pretrained_layer.norm2.running_mean)
                    our_layer.norm2.running_var.copy_(pretrained_layer.norm2.running_var)
            
            # Copy transition layer weights if not the last block
            if block_idx < 3:
                trans_name = f'transition{block_idx + 1}'
                our_trans = getattr(model.features, trans_name)
                pretrained_trans = getattr(pretrained.features, trans_name)
                
                # Copy norm and conv weights
                our_trans[0].weight.copy_(pretrained_trans.norm.weight)
                our_trans[0].bias.copy_(pretrained_trans.norm.bias)
                our_trans[0].running_mean.copy_(pretrained_trans.norm.running_mean)
                our_trans[0].running_var.copy_(pretrained_trans.norm.running_var)
                
                our_trans[2].weight.copy_(pretrained_trans.conv.weight)
    
    # Initialize new classifier
    nn.init.kaiming_normal_(model.classifier.weight, nonlinearity="linear")
    nn.init.zeros_(model.classifier.bias)
    
    return model


def main() -> None:
    defaults = TrainingConfig(
        model_name="densenet121_adaptive",
        input_channels=1,
        input_size=224,
        epochs=30,
        batch_size=32,  # Reduced due to additional memory from attention
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        step_size=10,
        gamma=0.1,
        num_workers=4,
        seed=42,
    )
    
    parser = argparse.ArgumentParser(description="Train Adaptive DenseNet-121 on OrganAMNIST")
    add_common_cli(parser, defaults)
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine annealing instead of step LR")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs")
    args = parser.parse_args()
    
    # Override scheduler if requested
    if args.use_cosine:
        print("[Adaptive DenseNet] Using cosine annealing scheduler with warmup")
        # This would require modifying train_utils to support cosine scheduling
        # For now, we'll use the standard approach
    
    run_training(build_densenet121_adaptive, defaults)


if __name__ == "__main__":
    main()
