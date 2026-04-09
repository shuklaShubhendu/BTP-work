"""
Fusion Module - Multi-Modal Cardiovascular Disease Detection
VisionCare: BTP Semester 7

Implements Intermediate Fusion strategy:
CXR (DenseNet-121) + ECG (1D-CNN) + Labs (MLP) → Fusion MLP → CVD Risk
"""

import torch
import torch.nn as nn


class ClinicalMLP(nn.Module):
    """MLP for blood lab values. Input: (B, 100), Output: (B, 2), (B, 64)"""
    
    def __init__(self, input_dim=100, num_classes=2):
        super().__init__()
        self.name = "MLP"
        self.feature_dim = 64
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_classes))
    
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features), features


class VisionCareFusion(nn.Module):
    """
    Multi-Modal Fusion for CVD Detection.
    
    Intermediate Fusion Strategy:
    - Extract features from each modality
    - Concatenate feature vectors
    - Learn cross-modal relationships via Fusion MLP
    
    Feature dimensions:
    - Vision (DenseNet-121): 1024
    - Signal (1D-CNN): 256
    - Clinical (MLP): 64
    - Total: 1344
    """
    
    def __init__(self, vision_model, signal_model, clinical_model, num_classes=2):
        super().__init__()
        self.vision = vision_model
        self.signal = signal_model
        self.clinical = clinical_model
        
        # Total concatenated features
        total_features = (
            vision_model.feature_dim +    # 1024 (DenseNet-121)
            signal_model.feature_dim +    # 256 (1D-CNN)
            clinical_model.feature_dim    # 64 (MLP)
        )  # = 1344
        
        # Fusion network - learns cross-modal relationships
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
        self.total_features = total_features
        print(f"✓ VisionCare Fusion initialized")
        print(f"  Vision: {vision_model.feature_dim}, Signal: {signal_model.feature_dim}, Clinical: {clinical_model.feature_dim}")
        print(f"  Total: {total_features} → {num_classes} classes")
    
    def forward(self, cxr, ecg, labs):
        """
        Forward pass.
        
        Args:
            cxr: (B, 3, 320, 320) Chest X-ray images
            ecg: (B, 12, 5000) ECG signals
            labs: (B, 100) Blood lab values
            
        Returns:
            logits: (B, 2) classification logits
            features: tuple of (vision_feat, signal_feat, clinical_feat)
        """
        # Extract features from each modality
        _, v_feat = self.vision(cxr)      # (B, 1024)
        _, s_feat = self.signal(ecg)      # (B, 256)
        _, c_feat = self.clinical(labs)   # (B, 64)
        
        # Concatenate features (Intermediate Fusion)
        combined = torch.cat([v_feat, s_feat, c_feat], dim=1)  # (B, 1344)
        
        # Fusion network
        logits = self.fusion(combined)  # (B, 2)
        
        return logits, (v_feat, s_feat, c_feat)
    
    def get_modality_predictions(self, cxr, ecg, labs):
        """Get predictions from each modality separately (for comparison)."""
        v_logits, v_feat = self.vision(cxr)
        s_logits, s_feat = self.signal(ecg)
        c_logits, c_feat = self.clinical(labs)
        return v_logits, s_logits, c_logits


class LateFusion(nn.Module):
    """
    Late Fusion alternative - average predictions.
    For comparison with Intermediate Fusion.
    """
    
    def __init__(self, vision_model, signal_model, clinical_model, weights=None):
        super().__init__()
        self.vision = vision_model
        self.signal = signal_model
        self.clinical = clinical_model
        
        # Weights for each modality (default: equal)
        if weights is None:
            weights = [1/3, 1/3, 1/3]
        self.weights = weights
        
        print(f"✓ Late Fusion initialized with weights: {weights}")
    
    def forward(self, cxr, ecg, labs):
        v_logits, _ = self.vision(cxr)
        s_logits, _ = self.signal(ecg)
        c_logits, _ = self.clinical(labs)
        
        # Weighted average of logits
        logits = (
            self.weights[0] * v_logits +
            self.weights[1] * s_logits +
            self.weights[2] * c_logits
        )
        
        return logits, (v_logits, s_logits, c_logits)


if __name__ == "__main__":
    from densenet_module import DenseNet121CXR
    from signal_module import ECG1DCNN
    
    print("Testing Fusion Module...")
    
    # Create models
    vision = DenseNet121CXR()
    signal = ECG1DCNN()
    clinical = ClinicalMLP()
    fusion = VisionCareFusion(vision, signal, clinical)
    
    # Test input
    cxr = torch.randn(2, 3, 320, 320)
    ecg = torch.randn(2, 12, 5000)
    labs = torch.randn(2, 100)
    
    logits, features = fusion(cxr, ecg, labs)
    print(f"\nFusion output: {logits.shape}")
    print(f"Feature shapes: {[f.shape for f in features]}")
    
    print("\n✓ Fusion module working!")
