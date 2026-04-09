"""
VisionCare - Multi-Modal MULTI-LABEL Training
SYMILE-MIMIC Dataset | 6 Disease Classification

Author: VisionCare Team - BTP Semester 7
Date: February 2026

PROJECT: Multi-Modal Medical Image Analysis
This script is the MAIN TRAINING PIPELINE for multi-label classification
using CXR (Chest X-Ray) + ECG + Labs (Blood Tests) modalities.

6 DISEASE LABELS (SYMILE-MIMIC):
=================================
Cardiomegaly, Edema, Atelectasis, Pleural Effusion, Lung Opacity, No Finding

USAGE:
======
# In Google Colab - Copy each section into different cells
# Each major section is marked with: # ===================== CELL X =====================

FEATURES:
=========
✅ Multi-label classification (6 diseases)
✅ Multi-modal fusion (CXR + ECG + Labs)
✅ Mixed precision training (FP16)
✅ Comprehensive visualizations
✅ Checkpointing for Colab disconnections
✅ Per-disease and macro/micro AUC metrics
"""


# ===================== CELL 1: SETUP & MOUNT DRIVE =====================
# Run this cell first to mount Google Drive and sync data

import os

# Check if running in Colab
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    # Mount Drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
        print("✅ Drive mounted!")
    
    # Configure AWS (only if needed)
    # !aws configure
    
    # Define destination
    DEST_DIR = "/content/drive/MyDrive/symile-mimic/"
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # THE CORRECT AWS SYNC COMMAND
    source_uri = "s3://arn:aws:s3:us-east-1:724665945834:accesspoint/symile-mimic-v1-0-0-01/symile-mimic/1.0.0/"
    
    # Uncomment below line to sync data (run once)
    # print(f"🚀 Syncing from PhysioNet Access Point to: {DEST_DIR}")
    # print("This might take 10-20 minutes...")
    # !aws s3 sync "{source_uri}" "{DEST_DIR}" --request-payer requester
    # print("✅ Sync Complete!")
else:
    DEST_DIR = "./data/symile-mimic/"
    print("Running locally, expecting data in:", DEST_DIR)


# ===================== CELL 2: IMPORTS & CONFIGURATION =====================

import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, 
    precision_recall_curve, multilabel_confusion_matrix
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

# Plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titleweight'] = 'bold'

print("✅ All libraries imported successfully!")


# ===================== CELL 3: CONFIGURATION =====================

class Config:
    """Training configuration."""
    
    # === PATHS (Adjust based on your environment) ===
    DATA_DIR = "/content/drive/MyDrive/symile-mimic" if IN_COLAB else "./data/symile-mimic"
    OUTPUT_DIR = "/content/drive/MyDrive/symile-mimic/MultiLabel_Results" if IN_COLAB else "MultiLabel_Results"
    CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
    
    # === MULTI-LABEL: 6 DISEASE CONDITIONS (SYMILE-MIMIC subset) ===
    LABELS = [
        'Cardiomegaly',      # Heart enlargement
        'Edema',             # Fluid in lungs
        'Atelectasis',       # Lung collapse
        'Pleural Effusion',  # Fluid around lungs
        'Lung Opacity',      # Abnormal density
        'No Finding'         # Normal/healthy
    ]
    NUM_LABELS = len(LABELS)  # 6
    
    # === TRAINING (modality-specific batch sizes) ===
    BATCH_SIZE_VISION = 32    # Large images need smaller batch
    BATCH_SIZE_SIGNAL = 64    # ECG is smaller, can use larger batch
    BATCH_SIZE_CLINICAL = 128 # Labs is tiny, can use large batch
    BATCH_SIZE_FUSION = 32    # Multi-modal needs smaller batch
    
    NUM_WORKERS = 0  # Use 0 to avoid multiprocessing warnings in Colab
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # === OPTIMIZATIONS ===
    USE_AMP = True  # Mixed precision (faster + less memory)
    GRAD_ACCUM_STEPS = 1
    LABEL_SMOOTHING = 0.1  # Regularization
    
    # === MODEL COMPARISON ===
    COMPARE_MODELS = True  # Train multiple models per modality and pick best
    
    # === DEVICE ===
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def print_config(cls):
        print(f"\n{'='*70}")
        print("⚙️  CONFIGURATION")
        print(f"{'='*70}")
        print(f"  Device: {cls.DEVICE}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"  Batch Sizes: Vision={cls.BATCH_SIZE_VISION}, Signal={cls.BATCH_SIZE_SIGNAL}, Clinical={cls.BATCH_SIZE_CLINICAL}")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Labels: {cls.NUM_LABELS} diseases")
        print(f"  Mixed Precision: {cls.USE_AMP}")
        print(f"  Model Comparison: {cls.COMPARE_MODELS}")
        print(f"  Output: {cls.OUTPUT_DIR}/")
        print(f"{'='*70}\n")

# Create output directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# Print config
Config.print_config()


# ===================== CELL 4: DATASET (MULTI-LABEL) =====================

class SymileMIMICMultiLabelDataset(Dataset):
    """
    Multi-Modal Dataset for SYMILE-MIMIC
    
    Modalities:
    - CXR: Chest X-Ray images (3, 320, 320)
    - ECG: 12-lead ECG waveforms (12, 5000)
    - Labs: Blood test values (50 percentiles + 50 missingness = 100)
    
    Labels: 6 disease conditions (multi-label)
    """
    
    def __init__(self, data_dir, split='train'):
        self.split = split
        self.data_dir = data_dir
        self.labels_list = Config.LABELS
        
        # Load CSV
        csv_path = f"{data_dir}/{split}.csv"
        print(f"  📂 Loading {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Load NPY files with memory mapping (efficient for large files)
        npy_dir = f"{data_dir}/data_npy/{split}"
        print(f"  📂 Loading NPY files from {npy_dir}...")
        
        self.cxr = np.load(f"{npy_dir}/cxr_{split}.npy", mmap_mode='r')
        self.ecg = np.load(f"{npy_dir}/ecg_{split}.npy", mmap_mode='r')
        self.labs_pct = np.load(f"{npy_dir}/labs_percentiles_{split}.npy", mmap_mode='r')
        self.labs_miss = np.load(f"{npy_dir}/labs_missingness_{split}.npy", mmap_mode='r')
        
        # === MULTI-LABEL EXTRACTION ===
        self.labels = self._extract_multilabels()
        
        # Print stats
        self._print_stats()
    
    def _extract_multilabels(self):
        """Extract 6 disease labels from CSV."""
        labels = np.zeros((len(self.df), Config.NUM_LABELS), dtype=np.float32)
        
        for i, col in enumerate(self.labels_list):
            if col in self.df.columns:
                values = self.df[col].fillna(0).values
                # Positive if 1.0 OR -1.0 (uncertain but leaning positive)
                labels[:, i] = ((values == 1.0) | (values == -1.0)).astype(float)
            else:
                print(f"  ⚠️ Warning: Column '{col}' not found in CSV")
        
        return labels
    
    def _print_stats(self):
        """Print dataset statistics."""
        print(f"\n  {'='*50}")
        print(f"  ✅ {self.split.upper()} Dataset Loaded")
        print(f"  {'='*50}")
        print(f"  Samples: {len(self):,}")
        print(f"  CXR shape: {self.cxr.shape}")
        print(f"  ECG shape: {self.ecg.shape}")
        print(f"  Labs shape: ({self.labs_pct.shape[1] + self.labs_miss.shape[1]},)")
        print(f"\n  📊 Label Distribution:")
        for i, name in enumerate(self.labels_list):
            count = int(self.labels[:, i].sum())
            pct = count / len(self) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {name:28s} {bar} {pct:5.1f}% ({count:,})")
        print(f"  {'='*50}\n")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # CXR: (3, 320, 320)
        cxr = torch.from_numpy(self.cxr[idx].copy()).float()
        
        # ECG: (12, 5000) 
        ecg = torch.from_numpy(self.ecg[idx].copy()).float()
        if ecg.dim() == 3:
            ecg = ecg.squeeze(0)
        if ecg.shape[0] != 12:
            ecg = ecg.transpose(0, 1)
        
        # Labs: (100,)
        labs = torch.from_numpy(
            np.concatenate([self.labs_pct[idx], self.labs_miss[idx]])
        ).float()
        
        # Labels: (6,)
        labels = torch.from_numpy(self.labels[idx]).float()
        
        return cxr, ecg, labs, labels


# ===================== CELL 5: VISION MODELS =====================

class DenseNet121MultiLabel(nn.Module):
    """DenseNet-121 - CheXNet baseline architecture."""
    
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "DenseNet-121"
        self.feature_dim = 1024
        self.params = "8M"
        
        # Pretrained backbone
        self.backbone = models.densenet121(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        
        # Multi-label classifier (6 outputs)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, num_labels)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


class EfficientNetB2MultiLabel(nn.Module):
    """EfficientNet-B2 - Efficient and accurate."""
    
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "EfficientNet-B2"
        self.feature_dim = 1408
        self.params = "9M"
        
        self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1408, num_labels)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


class ConvNeXtTinyMultiLabel(nn.Module):
    """ConvNeXt-Tiny - Modern CNN, competes with ViT."""
    
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "ConvNeXt-Tiny"
        self.feature_dim = 768
        self.params = "28M"
        
        self.backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(768),
            nn.Dropout(0.3),
            nn.Linear(768, num_labels)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.mean(dim=[-2, -1])
        logits = self.classifier(features)
        return logits, features


# ===================== CELL 6: SIGNAL MODELS (ECG) =====================

class CNN1DMultiLabel(nn.Module):
    """1D CNN for ECG - Fast baseline."""
    
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "1D-CNN"
        self.feature_dim = 256
        self.params = "0.5M"
        
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, 11, padding=5),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            
            nn.Conv1d(128, 256, 7, padding=3),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, x):
        features = self.conv(x).squeeze(-1)
        logits = self.classifier(features)
        return logits, features


class ResBlock1D(nn.Module):
    """Residual block for 1D signals."""
    
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch)
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1DMultiLabel(nn.Module):
    """ResNet-1D for ECG - Deep residual network."""
    
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "ResNet-1D"
        self.feature_dim = 256
        self.params = "2M"
        
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, 15, stride=2, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
    
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = [ResBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.avgpool(x).squeeze(-1)
        logits = self.classifier(features)
        return logits, features


class InceptionBlock(nn.Module):
    """Inception module with parallel multi-scale convolutions."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch), nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch), nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 7, padding=3),
            nn.BatchNorm1d(out_ch), nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 15, padding=7),
            nn.BatchNorm1d(out_ch), nn.ReLU()
        )
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.pool(out)


class InceptionTimeMultiLabel(nn.Module):
    """InceptionTime - SOTA for time-series classification."""
    
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "InceptionTime"
        self.feature_dim = 256
        self.params = "1.5M"
        
        self.inception1 = InceptionBlock(12, 32)   # -> 128 channels
        self.inception2 = InceptionBlock(128, 32)  # -> 128 channels
        self.inception3 = InceptionBlock(128, 32)  # -> 128 channels
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 256)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.avgpool(x).squeeze(-1)
        features = self.fc(x)
        logits = self.classifier(features)
        return logits, features


# ===================== CELL 7: CLINICAL MODEL (Labs) =====================

class MLPMultiLabel(nn.Module):
    """MLP for blood lab values."""
    
    def __init__(self, input_dim=100, num_labels=6):
        super().__init__()
        self.name = "MLP"
        self.feature_dim = 64
        self.params = "0.02M"
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_labels)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features


class TabNetMultiLabel(nn.Module):
    """TabNet - Attention-based feature selection for tabular data."""
    
    def __init__(self, input_dim=100, n_steps=3, hidden_dim=64, num_labels=6):
        super().__init__()
        self.name = "TabNet"
        self.feature_dim = 64
        self.params = "0.1M"
        self.n_steps = n_steps
        
        self.bn = nn.BatchNorm1d(input_dim)
        self.initial_fc = nn.Linear(input_dim, hidden_dim)
        
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.Softmax(dim=-1)
            ) for _ in range(n_steps)
        ])
        
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ) for _ in range(n_steps)
        ])
        
        self.final_fc = nn.Linear(hidden_dim, self.feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_labels)
        )
    
    def forward(self, x):
        x = self.bn(x)
        h = F.relu(self.initial_fc(x))
        
        aggregated = torch.zeros_like(h)
        for attn, fc in zip(self.attention_layers, self.fc_layers):
            mask = attn(h)
            masked = x * mask
            step_out = fc(masked)
            aggregated = aggregated + step_out
            h = step_out
        
        features = self.final_fc(aggregated)
        logits = self.classifier(features)
        return logits, features


# ===================== CELL 8: MULTI-MODAL FUSION =====================

class MultiModalFusionMultiLabel(nn.Module):
    """
    Multi-Modal Fusion Network
    
    Combines features from:
    - Vision (CXR): DenseNet/EfficientNet/ConvNeXt features
    - Signal (ECG): 1D-CNN/ResNet/InceptionTime features
    - Clinical (Labs): MLP/TabNet features
    
    Output: 6 disease predictions (SYMILE-MIMIC)
    """
    
    def __init__(self, vision_model, signal_model, clinical_model, num_labels=6):
        super().__init__()
        self.name = "MultiModal-Fusion"
        
        self.vision = vision_model
        self.signal = signal_model
        self.clinical = clinical_model
        
        # Calculate total feature dimension
        total_features = (
            vision_model.feature_dim + 
            signal_model.feature_dim + 
            clinical_model.feature_dim
        )
        self.feature_dim = total_features
        
        # Fusion network with attention
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.2),
            
            nn.Linear(128, num_labels)
        )
        
        print(f"\n  🔀 Fusion Network Created:")
        print(f"     Vision features:   {vision_model.feature_dim}")
        print(f"     Signal features:   {signal_model.feature_dim}")
        print(f"     Clinical features: {clinical_model.feature_dim}")
        print(f"     Total features:    {total_features}")
        print(f"     Output labels:     {num_labels}")
    
    def forward(self, cxr, ecg, labs):
        # Extract features from each modality
        _, vision_feat = self.vision(cxr)
        _, signal_feat = self.signal(ecg)
        _, clinical_feat = self.clinical(labs)
        
        # Concatenate
        combined = torch.cat([vision_feat, signal_feat, clinical_feat], dim=1)
        
        # Fuse
        logits = self.fusion(combined)
        
        return logits, (vision_feat, signal_feat, clinical_feat)


# ===================== CELL 9: MULTI-LABEL LOSS FUNCTION =====================

class MultiLabelBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss for Multi-Label Classification
    with optional label smoothing and class weighting.
    """
    
    def __init__(self, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # BCE with logits (numerically stable)
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        return loss


# ===================== CELL 10: METRICS & EVALUATION =====================

def compute_multilabel_metrics(labels, probs, threshold=0.5):
    """
    Compute comprehensive multi-label metrics.
    
    Returns:
        dict with macro/micro AUC, per-class AUC, F1, etc.
    """
    preds = (probs > threshold).astype(int)
    n_classes = labels.shape[1]
    
    results = {
        'per_class': {},
        'macro': {},
        'micro': {}
    }
    
    # Per-class metrics
    per_class_auc = []
    per_class_ap = []
    per_class_f1 = []
    
    for i in range(n_classes):
        class_name = Config.LABELS[i]
        y_true = labels[:, i]
        y_prob = probs[:, i]
        y_pred = preds[:, i]
        
        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            auc = 0.5
            ap = 0.0
        else:
            auc = roc_auc_score(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results['per_class'][class_name] = {
            'auc': auc,
            'ap': ap,
            'f1': f1,
            'support': int(y_true.sum())
        }
        
        per_class_auc.append(auc)
        per_class_ap.append(ap)
        per_class_f1.append(f1)
    
    # Macro metrics (average across classes)
    results['macro']['auc'] = np.mean(per_class_auc)
    results['macro']['ap'] = np.mean(per_class_ap)
    results['macro']['f1'] = np.mean(per_class_f1)
    
    # Micro metrics (aggregate predictions)
    try:
        results['micro']['auc'] = roc_auc_score(
            labels.ravel(), probs.ravel()
        )
    except ValueError:
        results['micro']['auc'] = 0.5
    
    results['micro']['f1'] = f1_score(
        labels.ravel(), preds.ravel(), zero_division=0
    )
    
    # Store arrays for plotting
    results['labels'] = labels
    results['probs'] = probs
    results['preds'] = preds
    
    return results


# ===================== CELL 11: TRAINER CLASS =====================

class MultiLabelTrainer:
    """
    Multi-Label Training Engine with:
    - Mixed precision (FP16)
    - Checkpointing
    - Early stopping
    - Comprehensive logging
    """
    
    def __init__(self, model, train_loader, val_loader, modality='fusion'):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.modality = modality
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS
        )
        
        # Loss function
        self.criterion = MultiLabelBCELoss(
            label_smoothing=Config.LABEL_SMOOTHING
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if Config.USE_AMP else None
        
        # History
        self.history = defaultdict(list)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (cxr, ecg, labs, labels) in enumerate(pbar):
            # Move to device
            cxr = cxr.to(Config.DEVICE, non_blocking=True)
            ecg = ecg.to(Config.DEVICE, non_blocking=True)
            labs = labs.to(Config.DEVICE, non_blocking=True)
            labels = labels.to(Config.DEVICE, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=Config.USE_AMP):
                if self.modality == 'fusion':
                    logits, _ = self.model(cxr, ecg, labs)
                elif self.modality == 'vision':
                    logits, _ = self.model(cxr)
                elif self.modality == 'signal':
                    logits, _ = self.model(ecg)
                else:  # clinical
                    logits, _ = self.model(labs)
                
                loss = self.criterion(logits, labels)
            
            # Backward pass
            if Config.USE_AMP:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        all_probs = []
        all_labels = []
        
        for cxr, ecg, labs, labels in tqdm(self.val_loader, desc="Evaluating", leave=False):
            cxr = cxr.to(Config.DEVICE, non_blocking=True)
            ecg = ecg.to(Config.DEVICE, non_blocking=True)
            labs = labs.to(Config.DEVICE, non_blocking=True)
            
            with autocast(enabled=Config.USE_AMP):
                if self.modality == 'fusion':
                    logits, _ = self.model(cxr, ecg, labs)
                elif self.modality == 'vision':
                    logits, _ = self.model(cxr)
                elif self.modality == 'signal':
                    logits, _ = self.model(ecg)
                else:
                    logits, _ = self.model(labs)
            
            probs = torch.sigmoid(logits.float())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        metrics = compute_multilabel_metrics(all_labels, all_probs)
        return metrics
    
    def train(self, epochs, save_name):
        """Full training loop with checkpointing."""
        print(f"\n{'='*70}")
        print(f"🚀 Training: {getattr(self.model, 'name', 'Model')}")
        print(f"   Modality: {self.modality.upper()}")
        print(f"   Epochs: {epochs}")
        print(f"{'='*70}")
        
        best_auc = 0
        best_epoch = 0
        patience_counter = 0
        patience = 5
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            metrics = self.evaluate()
            macro_auc = metrics['macro']['auc']
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_macro_auc'].append(macro_auc)
            self.history['val_micro_auc'].append(metrics['micro']['auc'])
            
            # Check for improvement
            marker = ""
            if macro_auc > best_auc:
                best_auc = macro_auc
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_auc': best_auc,
                    'metrics': metrics
                }
                torch.save(checkpoint, f"{Config.CHECKPOINT_DIR}/{save_name}.pth")
                marker = " ✅ Best! (saved)"
            else:
                patience_counter += 1
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Macro-AUC: {macro_auc:.4f} | "
                  f"Micro-AUC: {metrics['micro']['auc']:.4f} | "
                  f"Time: {epoch_time:.1f}s{marker}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        print(f"\n⏱️ Total training time: {total_time:.1f}s")
        print(f"🏆 Best Macro-AUC: {best_auc:.4f} (Epoch {best_epoch})")
        
        # Load best model (weights_only=False for PyTorch 2.6 compatibility)
        checkpoint = torch.load(f"{Config.CHECKPOINT_DIR}/{save_name}.pth", weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        final_metrics = self.evaluate()
        
        return {
            'model_name': getattr(self.model, 'name', 'Model'),
            'modality': self.modality,
            'feature_dim': getattr(self.model, 'feature_dim', 0),
            'params': getattr(self.model, 'params', 'N/A'),
            'history': dict(self.history),
            'best_auc': best_auc,
            'best_epoch': best_epoch,
            'train_time': total_time,
            'final_metrics': final_metrics
        }


# ===================== CELL 12: VISUALIZATION FUNCTIONS =====================

def plot_label_distribution(dataset, output_dir):
    """Plot distribution of 14 disease labels."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    counts = dataset.labels.sum(axis=0)
    percentages = counts / len(dataset) * 100
    
    colors = plt.cm.Spectral(np.linspace(0, 1, len(Config.LABELS)))
    bars = ax.bar(range(len(Config.LABELS)), percentages, color=colors, edgecolor='black')
    
    ax.set_xlabel('Disease Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Samples (%)', fontsize=12, fontweight='bold')
    ax.set_title('📊 Multi-Label Distribution: 14 Disease Conditions', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(Config.LABELS)))
    ax.set_xticklabels(Config.LABELS, rotation=45, ha='right')
    
    for bar, pct, cnt in zip(bars, percentages, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%\n({int(cnt):,})', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/label_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/label_distribution.png")
    plt.show()


def plot_per_class_auc(metrics, model_name, output_dir):
    """Plot per-class AUC-ROC scores."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    aucs = [metrics['per_class'][label]['auc'] for label in Config.LABELS]
    colors = plt.cm.RdYlGn(np.array(aucs))
    
    bars = ax.bar(range(len(Config.LABELS)), aucs, color=colors, edgecolor='black')
    
    ax.axhline(y=metrics['macro']['auc'], color='red', linestyle='--', 
               linewidth=2, label=f"Macro-AUC: {metrics['macro']['auc']:.3f}")
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Random (0.5)')
    
    ax.set_xlabel('Disease Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_title(f'📊 {model_name}: Per-Class AUC-ROC Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(Config.LABELS)))
    ax.set_xticklabels(Config.LABELS, rotation=45, ha='right')
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right')
    
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name.lower().replace(" ", "_")}_per_class_auc.png', 
                dpi=300, bbox_inches='tight')
    print(f"✅ Saved per-class AUC plot")
    plt.show()


def plot_roc_curves_multilabel(metrics, model_name, output_dir):
    """Plot ROC curves for all 14 classes."""
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    axes = axes.ravel()
    
    colors = plt.cm.tab20(np.linspace(0, 1, 14))
    
    for i, label in enumerate(Config.LABELS):
        ax = axes[i]
        y_true = metrics['labels'][:, i]
        y_prob = metrics['probs'][:, i]
        
        if len(np.unique(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = metrics['per_class'][label]['auc']
            ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=f'AUC={auc:.3f}')
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', transform=ax.transAxes)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'🔬 {model_name}: ROC Curves for 14 Diseases', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name.lower().replace(" ", "_")}_roc_curves.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ Saved ROC curves")
    plt.show()


def plot_training_history(history, model_name, output_dir):
    """Plot training curves (loss and AUC)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('📉 Training Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC
    axes[1].plot(history['val_macro_auc'], 'g-', linewidth=2, label='Macro-AUC')
    axes[1].plot(history['val_micro_auc'], 'b--', linewidth=2, label='Micro-AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title('📈 Validation AUC', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_name}: Training Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name.lower().replace(" ", "_")}_training_history.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ Saved training history")
    plt.show()


def plot_modality_comparison(results_list, output_dir):
    """Compare all modalities (Vision, Signal, Clinical, Fusion)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    names = [r['model_name'] for r in results_list]
    aucs = [r['best_auc'] for r in results_list]
    times = [r['train_time'] for r in results_list]
    
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63'][:len(names)]
    
    # AUC comparison
    bars = axes[0].bar(names, aucs, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Macro AUC-ROC', fontsize=12, fontweight='bold')
    axes[0].set_title('🏆 Multi-Modal Comparison: Macro-AUC', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0.5, max(aucs) + 0.1)
    
    # Highlight fusion
    if len(bars) >= 4:
        bars[-1].set_hatch('//')
    
    for bar, auc in zip(bars, aucs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', fontweight='bold', fontsize=11)
    
    # Training time
    axes[1].bar(names, times, color=colors, edgecolor='black')
    axes[1].set_ylabel('Training Time (s)', fontsize=12, fontweight='bold')
    axes[1].set_title('⏱️ Training Speed', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/modality_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/modality_comparison.png")
    plt.show()


def plot_fusion_improvement(all_results, output_dir):
    """Show fusion improvement over single modalities."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    modalities = ['Vision', 'Signal', 'Clinical', 'Fusion']
    aucs = [all_results[m]['best_auc'] for m in modalities]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    
    bars = ax.bar(modalities, aucs, color=colors, edgecolor='black', linewidth=2)
    
    # Annotate fusion improvement
    best_single = max(aucs[:3])
    fusion_auc = aucs[3]
    improvement = (fusion_auc - best_single) / best_single * 100
    
    ax.annotate(f'+{improvement:.1f}%\nImprovement',
                xy=(3, fusion_auc),
                xytext=(3, fusion_auc + 0.05),
                fontsize=12, fontweight='bold', color='green',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    ax.set_ylabel('Macro AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_title('🔀 Multi-Modal Fusion: Improvement Over Single Modalities',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, max(aucs) + 0.15)
    
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fusion_improvement.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/fusion_improvement.png")
    plt.show()


def plot_model_comparison_within_modality(vision_results, signal_results, clinical_results, output_dir):
    """Plot model comparison charts for each modality (shows all models tested)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Vision Models Comparison
    ax1 = axes[0]
    v_names = [r['model_name'] for r in vision_results]
    v_aucs = [r['best_auc'] for r in vision_results]
    colors = ['#2ecc71' if auc == max(v_aucs) else '#3498db' for auc in v_aucs]
    bars = ax1.bar(v_names, v_aucs, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Macro-AUC', fontsize=12, fontweight='bold')
    ax1.set_title('🩻 Vision Models', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.4, 0.9)
    for bar, auc in zip(bars, v_aucs):
        marker = " 🏆" if auc == max(v_aucs) else ""
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.4f}{marker}', ha='center', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    
    # Signal Models Comparison
    ax2 = axes[1]
    s_names = [r['model_name'] for r in signal_results]
    s_aucs = [r['best_auc'] for r in signal_results]
    colors = ['#2ecc71' if auc == max(s_aucs) else '#e74c3c' for auc in s_aucs]
    bars = ax2.bar(s_names, s_aucs, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Macro-AUC', fontsize=12, fontweight='bold')
    ax2.set_title('❤️ Signal Models (ECG)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.4, 0.9)
    for bar, auc in zip(bars, s_aucs):
        marker = " 🏆" if auc == max(s_aucs) else ""
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.4f}{marker}', ha='center', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    
    # Clinical Models Comparison
    ax3 = axes[2]
    c_names = [r['model_name'] for r in clinical_results]
    c_aucs = [r['best_auc'] for r in clinical_results]
    colors = ['#2ecc71' if auc == max(c_aucs) else '#9b59b6' for auc in c_aucs]
    bars = ax3.bar(c_names, c_aucs, color=colors, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Macro-AUC', fontsize=12, fontweight='bold')
    ax3.set_title('🧪 Clinical Models (Labs)', fontsize=14, fontweight='bold')
    ax3.set_ylim(0.4, 0.9)
    for bar, auc in zip(bars, c_aucs):
        marker = " 🏆" if auc == max(c_aucs) else ""
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.4f}{marker}', ha='center', fontsize=10, fontweight='bold')
    
    fig.suptitle('🔬 Model Comparison: Best vs Alternatives per Modality', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison_by_modality.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/model_comparison_by_modality.png")
    plt.show()


def plot_confusion_matrix_multilabel(metrics, model_name, output_dir, threshold=0.5):
    """Plot confusion matrices for all labels."""
    n_labels = len(Config.LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    probs = metrics['probs']
    labels = metrics['labels']
    preds = (probs > threshold).astype(int)
    
    for i, label in enumerate(Config.LABELS):
        ax = axes[i]
        
        # Calculate TP, TN, FP, FN
        tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum()
        tn = ((preds[:, i] == 0) & (labels[:, i] == 0)).sum()
        fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum()
        fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum()
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                    annot_kws={'size': 12, 'fontweight': 'bold'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
    
    fig.suptitle(f'📊 {model_name}: Confusion Matrices (Threshold=0.5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name.lower().replace(" ", "_")}_confusion_matrices.png', 
                dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion matrices")
    plt.show()


def plot_architecture_diagram(output_dir):
    """Create a visual architecture diagram of the multi-modal system."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    vision_color = '#3498db'
    signal_color = '#e74c3c'
    clinical_color = '#9b59b6'
    fusion_color = '#2ecc71'
    
    # Title
    ax.text(8, 9.5, 'VisionCare: Multi-Modal Deep Learning Architecture', 
            ha='center', fontsize=18, fontweight='bold')
    
    # Input boxes
    box_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black', linewidth=2)
    
    # Vision Input
    ax.add_patch(plt.Rectangle((0.5, 6.5), 3, 1.5, facecolor=vision_color, alpha=0.3, edgecolor='black', linewidth=2))
    ax.text(2, 7.5, '🩻 CXR Images\n(3×320×320)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Signal Input
    ax.add_patch(plt.Rectangle((0.5, 4), 3, 1.5, facecolor=signal_color, alpha=0.3, edgecolor='black', linewidth=2))
    ax.text(2, 5, '❤️ ECG Signals\n(12×5000)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Clinical Input
    ax.add_patch(plt.Rectangle((0.5, 1.5), 3, 1.5, facecolor=clinical_color, alpha=0.3, edgecolor='black', linewidth=2))
    ax.text(2, 2.5, '🧪 Lab Values\n(100 features)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Encoders
    ax.add_patch(plt.Rectangle((5, 6.5), 3.5, 1.5, facecolor=vision_color, alpha=0.6, edgecolor='black', linewidth=2))
    ax.text(6.75, 7.25, 'Vision Encoder\n(DenseNet/EfficientNet/\nConvNeXt)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.add_patch(plt.Rectangle((5, 4), 3.5, 1.5, facecolor=signal_color, alpha=0.6, edgecolor='black', linewidth=2))
    ax.text(6.75, 4.75, 'Signal Encoder\n(1D-CNN/ResNet-1D/\nInceptionTime)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.add_patch(plt.Rectangle((5, 1.5), 3.5, 1.5, facecolor=clinical_color, alpha=0.6, edgecolor='black', linewidth=2))
    ax.text(6.75, 2.25, 'Clinical Encoder\n(MLP/TabNet)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Feature vectors
    ax.add_patch(plt.Rectangle((10, 6.8), 1.5, 0.9, facecolor=vision_color, alpha=0.4, edgecolor='black'))
    ax.text(10.75, 7.25, '1024D', ha='center', va='center', fontsize=10)
    
    ax.add_patch(plt.Rectangle((10, 4.3), 1.5, 0.9, facecolor=signal_color, alpha=0.4, edgecolor='black'))
    ax.text(10.75, 4.75, '256D', ha='center', va='center', fontsize=10)
    
    ax.add_patch(plt.Rectangle((10, 1.8), 1.5, 0.9, facecolor=clinical_color, alpha=0.4, edgecolor='black'))
    ax.text(10.75, 2.25, '64D', ha='center', va='center', fontsize=10)
    
    # Fusion
    ax.add_patch(plt.Rectangle((12.5, 3.5), 2.5, 3, facecolor=fusion_color, alpha=0.6, edgecolor='black', linewidth=2))
    ax.text(13.75, 5, '🔀 Fusion\nNetwork\n\n512 → 256\n→ 128 → 6', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output
    ax.add_patch(plt.Rectangle((12.5, 0.5), 2.5, 1.5, facecolor='gold', alpha=0.6, edgecolor='black', linewidth=2))
    ax.text(13.75, 1.25, '🎯 6 Disease\nPredictions', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', color='black', lw=2)
    ax.annotate('', xy=(5, 7.25), xytext=(3.5, 7.25), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 4.75), xytext=(3.5, 4.75), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 2.25), xytext=(3.5, 2.25), arrowprops=arrow_props)
    
    ax.annotate('', xy=(10, 7.25), xytext=(8.5, 7.25), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 4.75), xytext=(8.5, 4.75), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 2.25), xytext=(8.5, 2.25), arrowprops=arrow_props)
    
    ax.annotate('', xy=(12.5, 5), xytext=(11.5, 7.25), arrowprops=arrow_props)
    ax.annotate('', xy=(12.5, 5), xytext=(11.5, 4.75), arrowprops=arrow_props)
    ax.annotate('', xy=(12.5, 5), xytext=(11.5, 2.25), arrowprops=arrow_props)
    
    ax.annotate('', xy=(13.75, 2), xytext=(13.75, 3.5), arrowprops=arrow_props)
    
    # Legend
    ax.text(8, 0.3, 'Inputs → Encoders → Feature Vectors → Fusion → Predictions', 
            ha='center', fontsize=12, style='italic')
    
    plt.savefig(f'{output_dir}/architecture_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✅ Saved: {output_dir}/architecture_diagram.png")
    plt.show()


def plot_per_disease_model_comparison(all_results, output_dir):
    """Compare AUC per disease across all modalities."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(Config.LABELS))
    width = 0.2
    
    modalities = ['Vision', 'Signal', 'Clinical', 'Fusion']
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
    
    for i, (mod, color) in enumerate(zip(modalities, colors)):
        per_class = all_results[mod]['final_metrics']['per_class']
        aucs = [per_class[label]['auc'] for label in Config.LABELS]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, aucs, width, label=mod, color=color, edgecolor='black', alpha=0.8)
    
    ax.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Disease', fontsize=12, fontweight='bold')
    ax.set_title('📊 Per-Disease AUC Comparison Across All Modalities', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(Config.LABELS, rotation=30, ha='right')
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Random')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_disease_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/per_disease_comparison.png")
    plt.show()


def plot_model_size_vs_performance(vision_results, signal_results, clinical_results, output_dir):
    """Scatter plot of model size vs performance."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect all results
    all_models = []
    colors = []
    markers = []
    
    for r in vision_results:
        all_models.append(r)
        colors.append('#3498db')
        markers.append('o')
    
    for r in signal_results:
        all_models.append(r)
        colors.append('#e74c3c')
        markers.append('s')
    
    for r in clinical_results:
        all_models.append(r)
        colors.append('#9b59b6')
        markers.append('^')
    
    # Parse param counts (e.g., "8M" -> 8)
    def parse_params(p):
        if isinstance(p, str):
            if 'M' in p:
                return float(p.replace('M', ''))
            elif 'K' in p:
                return float(p.replace('K', '')) / 1000
        return 1.0
    
    for i, (model, c, m) in enumerate(zip(all_models, colors, markers)):
        params = parse_params(model.get('params', '1M'))
        auc = model['best_auc']
        ax.scatter(params, auc, s=200, c=c, marker=m, edgecolors='black', linewidth=2)
        ax.annotate(model['model_name'], (params, auc), textcoords="offset points", 
                    xytext=(10, 5), fontsize=9)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=12, label='Vision'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c', markersize=12, label='Signal'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#9b59b6', markersize=12, label='Clinical'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    ax.set_xlabel('Model Size (Millions of Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro-AUC', fontsize=12, fontweight='bold')
    ax.set_title('📈 Model Size vs Performance Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 0.9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_size_vs_performance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/model_size_vs_performance.png")
    plt.show()


# ===================== CELL 13: REPORT GENERATION =====================


def generate_comprehensive_report(all_results, output_dir):
    """Generate detailed training report."""
    
    fusion = all_results['Fusion']
    
    report = f"""
{'='*80}
VISIONCARE - MULTI-MODAL MULTI-LABEL TRAINING REPORT
14 Disease Classification | SYMILE-MIMIC Dataset
BTP Semester 7 - February 2026
{'='*80}

📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
1. 📋 DATASET INFORMATION
{'='*80}
   Labels: 14 disease conditions
   Modalities: CXR (X-Ray) + ECG (12-lead) + Labs (Blood Tests)
   
   Disease Labels:
"""
    for i, label in enumerate(Config.LABELS):
        report += f"     {i+1:2d}. {label}\n"
    
    report += f"""
{'='*80}
2. 🩻 VISION MODULE (Chest X-Ray)
{'='*80}
   Model: {all_results['Vision']['model_name']}
   Parameters: {all_results['Vision']['params']}
   Feature Dimension: {all_results['Vision']['feature_dim']}
   Macro-AUC: {all_results['Vision']['best_auc']:.4f}
   Training Time: {all_results['Vision']['train_time']:.1f}s

{'='*80}
3. ❤️ SIGNAL MODULE (12-Lead ECG)
{'='*80}
   Model: {all_results['Signal']['model_name']}
   Parameters: {all_results['Signal']['params']}
   Feature Dimension: {all_results['Signal']['feature_dim']}
   Macro-AUC: {all_results['Signal']['best_auc']:.4f}
   Training Time: {all_results['Signal']['train_time']:.1f}s

{'='*80}
4. 🩸 CLINICAL MODULE (Blood Labs)
{'='*80}
   Model: {all_results['Clinical']['model_name']}
   Parameters: {all_results['Clinical']['params']}
   Feature Dimension: {all_results['Clinical']['feature_dim']}
   Macro-AUC: {all_results['Clinical']['best_auc']:.4f}
   Training Time: {all_results['Clinical']['train_time']:.1f}s

{'='*80}
5. 🔀 MULTI-MODAL FUSION
{'='*80}
   Combined Features: {fusion['feature_dim']}
   Macro-AUC: {fusion['best_auc']:.4f}
   Micro-AUC: {fusion['final_metrics']['micro']['auc']:.4f}
   Training Time: {fusion['train_time']:.1f}s

{'='*80}
6. 📊 FINAL COMPARISON
{'='*80}
   Modality        | Model              | Macro-AUC
   ----------------|--------------------|----------
   Vision (CXR)    | {all_results['Vision']['model_name']:18s} | {all_results['Vision']['best_auc']:.4f}
   Signal (ECG)    | {all_results['Signal']['model_name']:18s} | {all_results['Signal']['best_auc']:.4f}
   Clinical (Labs) | {all_results['Clinical']['model_name']:18s} | {all_results['Clinical']['best_auc']:.4f}
   ----------------|--------------------|----------
   🏆 FUSION       | Multi-Modal        | {fusion['best_auc']:.4f}

{'='*80}
7. 📈 FUSION IMPROVEMENT
{'='*80}
   Best Single Modality: {max(all_results[k]['best_auc'] for k in ['Vision', 'Signal', 'Clinical']):.4f}
   Fusion AUC: {fusion['best_auc']:.4f}
   Improvement: {((fusion['best_auc'] / max(all_results[k]['best_auc'] for k in ['Vision', 'Signal', 'Clinical'])) - 1) * 100:+.2f}%

   ✨ MULTI-MODAL FUSION CAPTURES COMPLEMENTARY INFORMATION! ✨

{'='*80}
8. 📁 FILES GENERATED
{'='*80}
   Visualizations:
     • label_distribution.png
     • *_per_class_auc.png
     • *_roc_curves.png
     • *_training_history.png
     • modality_comparison.png
     • fusion_improvement.png
   
   Models:
     • checkpoints/*.pth
   
   Reports:
     • training_report.txt
     • results.json

{'='*80}
✅ TRAINING COMPLETE - MULTI-MODAL MULTI-LABEL SUCCESS!
{'='*80}
"""
    
    print(report)
    
    with open(f'{output_dir}/training_report.txt', 'w') as f:
        f.write(report)
    print(f"✅ Saved: {output_dir}/training_report.txt")
    
    # Save JSON results
    json_results = {
        'labels': Config.LABELS,
        'modalities': {
            k: {
                'model': v['model_name'],
                'macro_auc': v['best_auc'],
                'train_time': v['train_time']
            } for k, v in all_results.items()
        },
        'per_class_auc': {
            label: all_results['Fusion']['final_metrics']['per_class'][label]['auc']
            for label in Config.LABELS
        }
    }
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✅ Saved: {output_dir}/results.json")


# ===================== CELL 14: MAIN TRAINING PIPELINE =====================

def main():
    """Main training pipeline with model comparison."""
    
    print("\n" + "="*70)
    print("🫀 VISIONCARE - MULTI-MODAL MULTI-LABEL TRAINING")
    print("   6 Disease Classification | SYMILE-MIMIC Dataset")
    print("   🔬 Model Comparison: Training multiple models per modality")
    print("="*70)
    
    Config.print_config()
    
    # ========================================
    # LOAD DATA
    # ========================================
    print("\n📂 Loading datasets...")
    train_ds = SymileMIMICMultiLabelDataset(Config.DATA_DIR, 'train')
    val_ds = SymileMIMICMultiLabelDataset(Config.DATA_DIR, 'val')
    
    # Plot label distribution
    plot_label_distribution(train_ds, Config.OUTPUT_DIR)
    
    # ========================================
    # 🩻 VISION MODULE - Compare 3 Models
    # ========================================
    print("\n" + "="*70)
    print("🩻 VISION MODULE - Comparing 3 Models")
    print("="*70)
    
    vision_loader_train = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE_VISION, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    vision_loader_val = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE_VISION, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    vision_models = [
        ('DenseNet-121', DenseNet121MultiLabel),
        ('EfficientNet-B2', EfficientNetB2MultiLabel),
        ('ConvNeXt-Tiny', ConvNeXtTinyMultiLabel),
    ]
    
    vision_results = []
    for name, ModelClass in vision_models:
        print(f"\n🔬 Training {name}...")
        model = ModelClass()
        trainer = MultiLabelTrainer(model, vision_loader_train, vision_loader_val, 'vision')
        result = trainer.train(Config.EPOCHS, f'vision_{name.lower().replace("-", "").replace(" ", "")}')
        vision_results.append(result)
        torch.cuda.empty_cache()
    
    # Find best vision model
    best_vision = max(vision_results, key=lambda x: x['best_auc'])
    print(f"\n📊 VISION MODEL COMPARISON:")
    for r in vision_results:
        marker = " 🏆 BEST" if r == best_vision else ""
        print(f"  {r['model_name']:20s} | Macro-AUC: {r['best_auc']:.4f}{marker}")
    
    plot_per_class_auc(best_vision['final_metrics'], 'Vision', Config.OUTPUT_DIR)
    plot_training_history(best_vision['history'], 'Vision', Config.OUTPUT_DIR)
    
    # ========================================
    # ❤️ SIGNAL MODULE - Compare 3 Models
    # ========================================
    print("\n" + "="*70)
    print("❤️ SIGNAL MODULE - Comparing 3 Models")
    print("="*70)
    
    signal_loader_train = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE_SIGNAL, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    signal_loader_val = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE_SIGNAL, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    signal_models = [
        ('1D-CNN', CNN1DMultiLabel),
        ('ResNet-1D', ResNet1DMultiLabel),
        ('InceptionTime', InceptionTimeMultiLabel),
    ]
    
    signal_results = []
    for name, ModelClass in signal_models:
        print(f"\n🔬 Training {name}...")
        model = ModelClass()
        trainer = MultiLabelTrainer(model, signal_loader_train, signal_loader_val, 'signal')
        result = trainer.train(Config.EPOCHS, f'signal_{name.lower().replace("-", "").replace(" ", "")}')
        signal_results.append(result)
        torch.cuda.empty_cache()
    
    # Find best signal model
    best_signal = max(signal_results, key=lambda x: x['best_auc'])
    print(f"\n📊 SIGNAL MODEL COMPARISON:")
    for r in signal_results:
        marker = " 🏆 BEST" if r == best_signal else ""
        print(f"  {r['model_name']:20s} | Macro-AUC: {r['best_auc']:.4f}{marker}")
    
    plot_per_class_auc(best_signal['final_metrics'], 'Signal', Config.OUTPUT_DIR)
    plot_training_history(best_signal['history'], 'Signal', Config.OUTPUT_DIR)
    
    # ========================================
    # 🩸 CLINICAL MODULE - Compare 2 Models
    # ========================================
    print("\n" + "="*70)
    print("🩸 CLINICAL MODULE - Comparing 2 Models")
    print("="*70)
    
    clinical_loader_train = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE_CLINICAL, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    clinical_loader_val = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE_CLINICAL, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    clinical_models = [
        ('MLP', MLPMultiLabel),
        ('TabNet', TabNetMultiLabel),
    ]
    
    clinical_results = []
    for name, ModelClass in clinical_models:
        print(f"\n🔬 Training {name}...")
        model = ModelClass()
        trainer = MultiLabelTrainer(model, clinical_loader_train, clinical_loader_val, 'clinical')
        result = trainer.train(Config.EPOCHS, f'clinical_{name.lower()}')
        clinical_results.append(result)
        torch.cuda.empty_cache()
    
    # Find best clinical model
    best_clinical = max(clinical_results, key=lambda x: x['best_auc'])
    print(f"\n📊 CLINICAL MODEL COMPARISON:")
    for r in clinical_results:
        marker = " 🏆 BEST" if r == best_clinical else ""
        print(f"  {r['model_name']:20s} | Macro-AUC: {r['best_auc']:.4f}{marker}")
    
    plot_per_class_auc(best_clinical['final_metrics'], 'Clinical', Config.OUTPUT_DIR)
    
    # ========================================
    # 🔀 MULTI-MODAL FUSION (using best models)
    # ========================================
    print("\n" + "="*70)
    print("🔀 MULTI-MODAL FUSION - Using Best Models")
    print(f"   Vision:   {best_vision['model_name']}")
    print(f"   Signal:   {best_signal['model_name']}")
    print(f"   Clinical: {best_clinical['model_name']}")
    print("="*70)
    
    fusion_loader_train = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE_FUSION, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    fusion_loader_val = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE_FUSION, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # Recreate best models for fusion
    vision_class = {'DenseNet-121': DenseNet121MultiLabel, 'EfficientNet-B2': EfficientNetB2MultiLabel, 'ConvNeXt-Tiny': ConvNeXtTinyMultiLabel}
    signal_class = {'1D-CNN': CNN1DMultiLabel, 'ResNet-1D': ResNet1DMultiLabel, 'InceptionTime': InceptionTimeMultiLabel}
    clinical_class = {'MLP': MLPMultiLabel, 'TabNet': TabNetMultiLabel}
    
    vision = vision_class[best_vision['model_name']]()
    signal = signal_class[best_signal['model_name']]()
    clinical = clinical_class[best_clinical['model_name']]()
    
    fusion_model = MultiModalFusionMultiLabel(vision, signal, clinical)
    trainer = MultiLabelTrainer(fusion_model, fusion_loader_train, fusion_loader_val, 'fusion')
    fusion_result = trainer.train(Config.EPOCHS, 'fusion_multimodal')
    
    plot_per_class_auc(fusion_result['final_metrics'], 'Fusion', Config.OUTPUT_DIR)
    plot_roc_curves_multilabel(fusion_result['final_metrics'], 'Fusion', Config.OUTPUT_DIR)
    plot_training_history(fusion_result['history'], 'Fusion', Config.OUTPUT_DIR)
    
    # ========================================
    # FINAL COMPARISON
    # ========================================
    all_results = {
        'Vision': best_vision,
        'Signal': best_signal,
        'Clinical': best_clinical,
        'Fusion': fusion_result
    }
    
    # Save model comparison results
    comparison = {
        'vision_models': [{'name': r['model_name'], 'auc': r['best_auc']} for r in vision_results],
        'signal_models': [{'name': r['model_name'], 'auc': r['best_auc']} for r in signal_results],
        'clinical_models': [{'name': r['model_name'], 'auc': r['best_auc']} for r in clinical_results],
        'best_models': {
            'vision': best_vision['model_name'],
            'signal': best_signal['model_name'],
            'clinical': best_clinical['model_name']
        }
    }
    
    with open(f"{Config.OUTPUT_DIR}/model_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # ========================================
    # 📊 GENERATE ALL VISUALIZATIONS
    # ========================================
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS...")
    print("="*70)
    
    # Architecture diagram (generate first!)
    plot_architecture_diagram(Config.OUTPUT_DIR)
    
    # Model comparison within each modality
    plot_model_comparison_within_modality(vision_results, signal_results, clinical_results, Config.OUTPUT_DIR)
    
    # Model size vs performance trade-off
    plot_model_size_vs_performance(vision_results, signal_results, clinical_results, Config.OUTPUT_DIR)
    
    # Modality comparison (best models)
    plot_modality_comparison(list(all_results.values()), Config.OUTPUT_DIR)
    
    # Fusion improvement chart
    plot_fusion_improvement(all_results, Config.OUTPUT_DIR)
    
    # Per-disease comparison across modalities
    plot_per_disease_model_comparison(all_results, Config.OUTPUT_DIR)
    
    # Confusion matrices for fusion model
    plot_confusion_matrix_multilabel(fusion_result['final_metrics'], 'Fusion', Config.OUTPUT_DIR)
    
    # Generate comprehensive report
    generate_comprehensive_report(all_results, Config.OUTPUT_DIR)

    
    # ========================================
    # DONE!
    # ========================================
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    
    print("\n� FINAL MODEL COMPARISON:")
    print(f"  {'Modality':<12} | {'Best Model':<20} | {'Macro-AUC':>10}")
    print(f"  {'-'*12}-+-{'-'*20}-+-{'-'*10}")
    print(f"  {'Vision':<12} | {best_vision['model_name']:<20} | {best_vision['best_auc']:>10.4f}")
    print(f"  {'Signal':<12} | {best_signal['model_name']:<20} | {best_signal['best_auc']:>10.4f}")
    print(f"  {'Clinical':<12} | {best_clinical['model_name']:<20} | {best_clinical['best_auc']:>10.4f}")
    print(f"  {'-'*12}-+-{'-'*20}-+-{'-'*10}")
    print(f"  {'🔀 FUSION':<12} | {'Multi-Modal':<20} | {fusion_result['best_auc']:>10.4f} 🏆")
    
    improvement = fusion_result['best_auc'] - max(best_vision['best_auc'], best_signal['best_auc'], best_clinical['best_auc'])
    print(f"\n  📈 Fusion Improvement: +{improvement:.4f} over best single modality")
    
    print(f"\n📁 Results saved to: {Config.OUTPUT_DIR}/")
    print("✨ MULTI-MODAL MULTI-LABEL SUCCESS! ✨")


# ===================== CELL 15: RUN TRAINING =====================

if __name__ == "__main__":
    main()
