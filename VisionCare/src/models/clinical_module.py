"""
VisionCare - Clinical Module (Tabular Data Classification)
==========================================================
MLP-based model for detecting cardiovascular conditions from clinical data.
"""

import torch
import torch.nn as nn
import numpy as np


class ClinicalModule(nn.Module):
    """
    Clinical module for tabular data based CVD detection.
    
    Uses Multi-Layer Perceptron to process clinical features like:
    - Demographics (age, gender)
    - Vital signs (heart rate, blood pressure)
    - Lab values (troponin, BNP)
    - Medical history (diabetes, hypertension)
    
    Input: [batch, num_features] - normalized clinical features
    Output:
        - logits: [batch, num_classes]
        - features: [batch, 64] for fusion layer
    """
    
    def __init__(self, input_features=50, num_classes=2, hidden_dims=[128, 64]):
        """
        Args:
            input_features: Number of clinical features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.input_features = input_features
        self.feature_dim = hidden_dims[-1]  # Last hidden layer for fusion
        
        # Build feature extractor
        layers = []
        prev_dim = input_features
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input clinical features [batch, num_features]
            
        Returns:
            logits: Class predictions [batch, num_classes]
            features: Feature embeddings [batch, 64] for fusion
        """
        features = self.feature_extractor(x)  # [batch, 64]
        logits = self.classifier(features)    # [batch, num_classes]
        
        return logits, features
    
    def predict_proba(self, x):
        """Get probability predictions."""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def get_feature_dim(self):
        """Return the feature dimension for fusion layer."""
        return self.feature_dim


class ClinicalFeatureConfig:
    """
    Configuration for clinical features to extract from MIMIC-IV.
    
    This defines which features to extract and how to normalize them.
    """
    
    # Demographic features
    DEMOGRAPHIC_FEATURES = [
        'anchor_age',      # Patient age
        'gender',          # M=1, F=0
    ]
    
    # Vital signs (from chartevents)
    VITAL_FEATURES = {
        'heart_rate': {'itemid': [220045], 'normal_range': (60, 100)},
        'sbp': {'itemid': [220050, 220179], 'normal_range': (90, 140)},  # Systolic BP
        'dbp': {'itemid': [220051, 220180], 'normal_range': (60, 90)},   # Diastolic BP
        'resp_rate': {'itemid': [220210], 'normal_range': (12, 20)},
        'spo2': {'itemid': [220277], 'normal_range': (95, 100)},
        'temperature': {'itemid': [223761, 223762], 'normal_range': (36, 38)},
    }
    
    # Lab values (from labevents) - CVD relevant
    LAB_FEATURES = {
        'troponin_t': {'itemid': [51003], 'normal_range': (0, 0.04)},     # Heart damage marker
        'bnp': {'itemid': [50963], 'normal_range': (0, 100)},             # Heart failure marker
        'creatinine': {'itemid': [50912], 'normal_range': (0.7, 1.3)},    # Kidney function
        'potassium': {'itemid': [50971], 'normal_range': (3.5, 5.0)},     # Electrolyte
        'sodium': {'itemid': [50983], 'normal_range': (136, 145)},        # Electrolyte
        'glucose': {'itemid': [50931], 'normal_range': (70, 100)},        # Blood sugar
        'hemoglobin': {'itemid': [51222], 'normal_range': (12, 17)},      # Anemia indicator
        'wbc': {'itemid': [51301], 'normal_range': (4, 11)},              # Infection marker
        'platelets': {'itemid': [51265], 'normal_range': (150, 400)},
        'cholesterol': {'itemid': [50907], 'normal_range': (0, 200)},     # CVD risk
    }
    
    # Binary comorbidity flags (from diagnoses)
    COMORBIDITY_FEATURES = [
        'has_hypertension',    # I10-I15
        'has_diabetes',        # E10-E14
        'has_obesity',         # E66
        'has_smoking_history', # F17, Z87.891
        'has_prev_mi',         # I21-I22 history
        'has_atrial_fib',      # I48
        'has_chf_history',     # I50 history
        'has_ckd',             # N18
    ]
    
    @classmethod
    def get_total_features(cls):
        return (
            len(cls.DEMOGRAPHIC_FEATURES) + 
            len(cls.VITAL_FEATURES) + 
            len(cls.LAB_FEATURES) + 
            len(cls.COMORBIDITY_FEATURES)
        )


def prepare_clinical_features(patient_data: dict) -> torch.Tensor:
    """
    Prepare clinical features from raw MIMIC-IV data.
    
    Args:
        patient_data: Dictionary containing patient's clinical data
        
    Returns:
        Normalized feature tensor [1, num_features]
    """
    features = []
    
    # Demographics
    features.append(patient_data.get('anchor_age', 65) / 100.0)  # Normalize age
    features.append(1.0 if patient_data.get('gender', 'M') == 'M' else 0.0)
    
    # Vitals (use mean if available, else normal value)
    for name, config in ClinicalFeatureConfig.VITAL_FEATURES.items():
        value = patient_data.get(name, np.mean(config['normal_range']))
        normalized = (value - config['normal_range'][0]) / (
            config['normal_range'][1] - config['normal_range'][0]
        )
        features.append(np.clip(normalized, -1, 2))  # Allow some range beyond normal
    
    # Labs
    for name, config in ClinicalFeatureConfig.LAB_FEATURES.items():
        value = patient_data.get(name, np.mean(config['normal_range']))
        normalized = (value - config['normal_range'][0]) / (
            config['normal_range'][1] - config['normal_range'][0]
        )
        features.append(np.clip(normalized, -1, 2))
    
    # Comorbidities (binary flags)
    for name in ClinicalFeatureConfig.COMORBIDITY_FEATURES:
        features.append(float(patient_data.get(name, False)))
    
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


# Example usage and testing
if __name__ == "__main__":
    # Get total features
    num_features = ClinicalFeatureConfig.get_total_features()
    print(f"Total clinical features: {num_features}")
    
    # Test the model
    model = ClinicalModule(input_features=num_features, num_classes=2)
    
    # Dummy input (batch of 4 patients)
    dummy_input = torch.randn(4, num_features)
    
    # Forward pass
    logits, features = model(dummy_input)
    
    print(f"\nModel: ClinicalModule (MLP)")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Feature shape for fusion: {features.shape}")
    print(f"Feature dimension: {model.get_feature_dim()}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test with example patient data
    print("\n--- Example Patient Processing ---")
    patient = {
        'anchor_age': 72,
        'gender': 'M',
        'heart_rate': 88,
        'sbp': 145,
        'troponin_t': 0.08,  # Elevated
        'has_hypertension': True,
        'has_diabetes': True,
    }
    
    features = prepare_clinical_features(patient)
    print(f"Prepared feature vector shape: {features.shape}")
