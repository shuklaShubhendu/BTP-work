"""
VisionCare - Fusion Model
=========================
Combines Vision (X-ray), Signal (ECG), and Clinical features for CVD prediction.
This is the heart of the VisionCare project!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_module import VisionModule
from .signal_module import SignalModule
from .clinical_module import ClinicalModule


class VisionCareFusion(nn.Module):
    """
    Multi-modal fusion model for cardiovascular disease detection.
    
    Combines features from three modalities:
    - Vision: Chest X-ray features (2048-dim from ResNet-50)
    - Signal: ECG waveform features (256-dim from 1D-CNN)
    - Clinical: Tabular clinical features (64-dim from MLP)
    
    Total concatenated features: 2048 + 256 + 64 = 2368
    
    Fusion Strategy: Feature concatenation + learned fusion layers
    """
    
    def __init__(
        self,
        num_classes=2,
        clinical_input_features=28,
        freeze_individual_modules=True,
        fusion_hidden_dims=[512, 128],
        dropout=0.4
    ):
        """
        Args:
            num_classes: Number of output classes (2 for binary CVD detection)
            clinical_input_features: Number of clinical features
            freeze_individual_modules: Whether to freeze pretrained modules
            fusion_hidden_dims: Hidden layer dimensions for fusion network
            dropout: Dropout rate in fusion layers
        """
        super().__init__()
        
        # Individual modality modules
        self.vision_module = VisionModule(num_classes=num_classes, pretrained=True)
        self.signal_module = SignalModule(num_classes=num_classes)
        self.clinical_module = ClinicalModule(
            input_features=clinical_input_features,
            num_classes=num_classes
        )
        
        # Get feature dimensions from each module
        self.vision_dim = self.vision_module.get_feature_dim()    # 2048
        self.signal_dim = self.signal_module.get_feature_dim()    # 256
        self.clinical_dim = self.clinical_module.get_feature_dim() # 64
        self.total_dim = self.vision_dim + self.signal_dim + self.clinical_dim
        
        # Optionally freeze individual modules
        if freeze_individual_modules:
            self._freeze_modules()
        
        # Fusion network
        fusion_layers = []
        prev_dim = self.total_dim
        
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        fusion_layers.append(nn.Linear(prev_dim, num_classes))
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        # Learnable modality weights (for interpretability)
        self.modality_attention = nn.Parameter(torch.ones(3) / 3)
    
    def _freeze_modules(self):
        """Freeze parameters of individual modality modules."""
        for param in self.vision_module.parameters():
            param.requires_grad = False
        for param in self.signal_module.parameters():
            param.requires_grad = False
        for param in self.clinical_module.parameters():
            param.requires_grad = False
    
    def load_pretrained_modules(self, vision_path=None, signal_path=None, clinical_path=None):
        """
        Load pretrained weights for individual modules.
        
        Args:
            vision_path: Path to vision module checkpoint
            signal_path: Path to signal module checkpoint
            clinical_path: Path to clinical module checkpoint
        """
        if vision_path:
            self.vision_module.load_state_dict(torch.load(vision_path))
            print(f"✅ Loaded vision module from {vision_path}")
        
        if signal_path:
            self.signal_module.load_state_dict(torch.load(signal_path))
            print(f"✅ Loaded signal module from {signal_path}")
        
        if clinical_path:
            self.clinical_module.load_state_dict(torch.load(clinical_path))
            print(f"✅ Loaded clinical module from {clinical_path}")
    
    def forward(self, xray_image, ecg_signal, clinical_data):
        """
        Forward pass through all modalities and fusion.
        
        Args:
            xray_image: Chest X-ray [batch, 3, 224, 224]
            ecg_signal: ECG waveform [batch, 12, 5000]
            clinical_data: Clinical features [batch, num_features]
            
        Returns:
            logits: Class predictions [batch, num_classes]
            modality_weights: Learned attention weights for each modality
        """
        # Extract features from each modality
        _, vision_features = self.vision_module(xray_image)     # [batch, 2048]
        _, signal_features = self.signal_module(ecg_signal)     # [batch, 256]
        _, clinical_features = self.clinical_module(clinical_data)  # [batch, 64]
        
        # Apply attention-weighted scaling (optional)
        weights = F.softmax(self.modality_attention, dim=0)
        
        # Concatenate all features
        combined = torch.cat([
            vision_features * weights[0],
            signal_features * weights[1],
            clinical_features * weights[2]
        ], dim=1)  # [batch, 2368]
        
        # Fusion network
        logits = self.fusion_network(combined)  # [batch, num_classes]
        
        return logits, weights
    
    def predict(self, xray_image, ecg_signal, clinical_data):
        """Get probability predictions and risk score."""
        logits, weights = self.forward(xray_image, ecg_signal, clinical_data)
        probs = F.softmax(logits, dim=1)
        
        # Risk score (probability of CVD)
        risk_score = probs[:, 1]  # Probability of positive class
        
        return {
            'risk_score': risk_score,
            'probabilities': probs,
            'modality_weights': weights,
            'logits': logits
        }


class WeightedProbabilityFusion(nn.Module):
    """
    Simpler late fusion approach: weighted average of individual model probabilities.
    
    This approach:
    1. Gets probability predictions from each pretrained module
    2. Combines them using learnable weights
    
    Pros: Simple, interpretable, works well with small fusion datasets
    Cons: Less powerful than feature-level fusion
    """
    
    def __init__(self, num_classes=2, initial_weights=None):
        super().__init__()
        
        # Learnable weights for each modality
        if initial_weights is None:
            initial_weights = [0.4, 0.4, 0.2]  # Vision, Signal, Clinical
        
        self.weights = nn.Parameter(torch.tensor(initial_weights))
    
    def forward(self, p_vision, p_signal, p_clinical):
        """
        Combine probability predictions from individual models.
        
        Args:
            p_vision: Vision module probabilities [batch, num_classes]
            p_signal: Signal module probabilities [batch, num_classes]
            p_clinical: Clinical module probabilities [batch, num_classes]
            
        Returns:
            Fused probabilities [batch, num_classes]
        """
        # Normalize weights to sum to 1
        w = F.softmax(self.weights, dim=0)
        
        # Weighted combination
        fused = w[0] * p_vision + w[1] * p_signal + w[2] * p_clinical
        
        return fused, w


class AttentionFusion(nn.Module):
    """
    Attention-based fusion: dynamically weight modalities based on input.
    
    For each sample, compute attention weights based on the features themselves,
    allowing the model to emphasize different modalities for different patients.
    """
    
    def __init__(self, vision_dim=2048, signal_dim=256, clinical_dim=64, num_classes=2):
        super().__init__()
        
        # Project each modality to same dimension
        hidden_dim = 256
        
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.signal_proj = nn.Linear(signal_dim, hidden_dim)
        self.clinical_proj = nn.Linear(clinical_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.Tanh(),
            nn.Linear(64, 3)  # 3 modalities
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, vision_feat, signal_feat, clinical_feat):
        """
        Attention-weighted fusion.
        
        Args:
            vision_feat: [batch, vision_dim]
            signal_feat: [batch, signal_dim]
            clinical_feat: [batch, clinical_dim]
        """
        # Project to common space
        v = self.vision_proj(vision_feat)    # [batch, hidden]
        s = self.signal_proj(signal_feat)    # [batch, hidden]
        c = self.clinical_proj(clinical_feat) # [batch, hidden]
        
        # Compute attention weights
        combined = torch.cat([v, s, c], dim=1)  # [batch, hidden*3]
        attention_weights = F.softmax(self.attention(combined), dim=1)  # [batch, 3]
        
        # Weighted sum
        fused = (
            attention_weights[:, 0:1] * v +
            attention_weights[:, 1:2] * s +
            attention_weights[:, 2:3] * c
        )  # [batch, hidden]
        
        # Classify
        logits = self.classifier(fused)
        
        return logits, attention_weights


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("🫀 VisionCare Fusion Models")
    print("="*60)
    
    # Test VisionCareFusion
    model = VisionCareFusion(num_classes=2, clinical_input_features=28)
    
    # Dummy inputs
    batch_size = 4
    xray = torch.randn(batch_size, 3, 224, 224)
    ecg = torch.randn(batch_size, 12, 5000)
    clinical = torch.randn(batch_size, 28)
    
    # Forward pass
    logits, weights = model(xray, ecg, clinical)
    
    print(f"\n📊 VisionCareFusion Model Summary:")
    print(f"   Vision features:   {model.vision_dim}")
    print(f"   Signal features:   {model.signal_dim}")
    print(f"   Clinical features: {model.clinical_dim}")
    print(f"   Total features:    {model.total_dim}")
    print(f"\n   Output logits shape: {logits.shape}")
    print(f"   Modality weights: {weights.detach().numpy()}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test prediction
    result = model.predict(xray, ecg, clinical)
    print(f"\n🎯 Sample Prediction:")
    print(f"   Risk scores: {result['risk_score'].detach().numpy()}")
