"""
VisionCare - Vision Module (Chest X-Ray Classification)
=======================================================
ResNet-50 based model for detecting cardiovascular conditions from X-rays.
"""

import torch
import torch.nn as nn
from torchvision import models


class VisionModule(nn.Module):
    """
    Vision module for chest X-ray based CVD detection.
    
    Uses pretrained ResNet-50 as backbone with custom classification head.
    Returns both predictions and feature embeddings for fusion.
    
    Input: [batch, 3, 224, 224] - RGB chest X-ray images
    Output: 
        - logits: [batch, num_classes]
        - features: [batch, 2048] for fusion layer
    """
    
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Number of output classes (default: 2 for binary CVD detection)
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze ResNet layers (useful for fusion training)
        """
        super().__init__()
        
        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Get feature dimension before removing FC layer
        self.feature_dim = self.backbone.fc.in_features  # 2048
        
        # Remove the original fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone layers for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, unfreeze_from='layer4'):
        """
        Gradually unfreeze backbone layers for fine-tuning.
        
        Args:
            unfreeze_from: Which layer to start unfreezing from
                          Options: 'layer1', 'layer2', 'layer3', 'layer4'
        """
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        start_idx = layer_names.index(unfreeze_from)
        
        for name, param in self.backbone.named_parameters():
            for layer in layer_names[start_idx:]:
                if layer in name:
                    param.requires_grad = True
                    break
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [batch, 3, 224, 224]
            
        Returns:
            logits: Class predictions [batch, num_classes]
            features: Feature embeddings [batch, 2048] for fusion
        """
        # Extract features from backbone
        features = self.backbone(x)  # [batch, 2048]
        
        # Classification
        logits = self.classifier(features)  # [batch, num_classes]
        
        return logits, features
    
    def predict_proba(self, x):
        """Get probability predictions."""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def get_feature_dim(self):
        """Return the feature dimension for fusion layer."""
        return self.feature_dim


class DenseNetVisionModule(nn.Module):
    """
    Alternative vision module using DenseNet-121.
    Often performs better on medical imaging tasks.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        
        self.feature_dim = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


# Example usage and testing
if __name__ == "__main__":
    # Test the model
    model = VisionModule(num_classes=2, pretrained=True)
    
    # Dummy input (batch of 4 images)
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # Forward pass
    logits, features = model(dummy_input)
    
    print(f"Model: VisionModule (ResNet-50)")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Feature shape for fusion: {features.shape}")
    print(f"Feature dimension: {model.get_feature_dim()}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
