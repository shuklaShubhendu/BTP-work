"""
Vision Module - DenseNet-121 for CXR Analysis
VisionCare: Cardiovascular Disease Detection
Project: BTP Semester 7

DenseNet-121 is preferred over ResNet-50 for medical imaging because:
1. Fewer parameters (8M vs 25M) - less overfitting
2. Dense connections - better feature reuse
3. Proven on CheXpert/CheXNet (Stanford)
"""

import torch
import torch.nn as nn
from torchvision import models


class DenseNet121CXR(nn.Module):
    """
    DenseNet-121 for Chest X-Ray classification.
    
    Pretrained on ImageNet, fine-tuned for Cardiomegaly detection.
    Used in CheXNet (Stanford) for multi-label classification.
    
    Input:  (B, 3, 320, 320) - CXR images
    Output: (B, 2) logits, (B, 1024) features
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.name = "DenseNet-121"
        self.feature_dim = 1024  # DenseNet-121 feature dimension
        
        # Load pretrained DenseNet-121
        if pretrained:
            self.backbone = models.densenet121(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.densenet121(weights=None)
        
        # Replace classifier
        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
        print(f"✓ {self.name} initialized (features: {self.feature_dim})")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, 3, 320, 320) CXR images
            
        Returns:
            logits: (B, 2) classification logits
            features: (B, 1024) extracted features for fusion
        """
        features = self.backbone(x)  # (B, 1024)
        logits = self.classifier(features)  # (B, 2)
        return logits, features
    
    def get_features(self, x):
        """Extract features only (for fusion)."""
        return self.backbone(x)


class ResNet50CXR(nn.Module):
    """
    ResNet-50 alternative for comparison.
    
    Input:  (B, 3, 320, 320) - CXR images
    Output: (B, 2) logits, (B, 2048) features
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.name = "ResNet-50"
        self.feature_dim = 2048
        
        if pretrained:
            self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        else:
            self.backbone = models.resnet50(weights=None)
        
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)
        )
        
        print(f"✓ {self.name} initialized (features: {self.feature_dim})")
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


class EfficientNetB0CXR(nn.Module):
    """
    EfficientNet-B0 for efficient inference.
    
    Input:  (B, 3, 320, 320) - CXR images
    Output: (B, 2) logits, (B, 1280) features
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.name = "EfficientNet-B0"
        self.feature_dim = 1280
        
        if pretrained:
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes)
        )
        
        print(f"✓ {self.name} initialized (features: {self.feature_dim})")
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


# Model factory
def get_vision_model(model_name='densenet121', num_classes=2, pretrained=True):
    """
    Factory function to get vision model.
    
    Args:
        model_name: 'densenet121', 'resnet50', or 'efficientnet'
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        Model instance
    """
    models_dict = {
        'densenet121': DenseNet121CXR,
        'resnet50': ResNet50CXR,
        'efficientnet': EfficientNetB0CXR
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models_dict.keys())}")
    
    return models_dict[model_name](num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    # Test all models
    print("Testing Vision Models...")
    
    x = torch.randn(2, 3, 320, 320)
    
    for name in ['densenet121', 'resnet50', 'efficientnet']:
        model = get_vision_model(name)
        logits, features = model(x)
        print(f"  {name}: logits={logits.shape}, features={features.shape}")
    
    print("\n✓ All models working!")
