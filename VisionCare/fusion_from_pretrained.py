# =====================================================================
# 🔀 VISIONCARE: FUSION TRAINING FROM PRE-TRAINED MODELS
# =====================================================================
# This script loads the best pre-trained models from Google Drive
# and trains ONLY the fusion model (no retraining of modality encoders)
# 
# Features:
# - Loads checkpoints from Drive
# - Comprehensive metrics (Accuracy, Precision, Recall, F1, Specificity)
# - Freezes encoder weights (optional)
# - Generates detailed reports
# =====================================================================

# ===================== CELL 1: IMPORTS =====================

import os
import sys
import json
import time
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import torchvision.models as models
from tqdm.auto import tqdm

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')

# ===================== CELL 2: ENVIRONMENT DETECTION =====================

try:
    from google.colab import drive
    IN_COLAB = True
    drive.mount('/content/drive')
    print("✅ Running in Google Colab - Drive mounted!")
except:
    IN_COLAB = False
    print("💻 Running locally")


# ===================== CELL 3: CONFIGURATION =====================

class Config:
    """Training configuration."""
    
    # === PATHS ===
    DATA_DIR = "/content/drive/MyDrive/symile-mimic" if IN_COLAB else "./data/symile-mimic"
    OUTPUT_DIR = "/content/drive/MyDrive/symile-mimic/MultiLabel_Results" if IN_COLAB else "MultiLabel_Results"
    CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
    
    # === LABELS ===
    LABELS = [
        'Cardiomegaly', 'Edema', 'Atelectasis', 
        'Pleural Effusion', 'Lung Opacity', 'No Finding'
    ]
    NUM_LABELS = len(LABELS)
    
    # === BEST MODELS (from your training results) ===
    BEST_VISION_MODEL = 'ConvNeXt-Tiny'      # 🏆 0.7694 AUC
    BEST_SIGNAL_MODEL = '1D-CNN'              # 🏆 0.6023 AUC
    BEST_CLINICAL_MODEL = 'MLP'               # 🏆 0.6118 AUC
    
    # === CHECKPOINT FILES ===
    VISION_CHECKPOINT = f"{CHECKPOINT_DIR}/vision_convnexttiny.pth"
    SIGNAL_CHECKPOINT = f"{CHECKPOINT_DIR}/signal_1dcnn.pth"  
    CLINICAL_CHECKPOINT = f"{CHECKPOINT_DIR}/clinical_mlp.pth"
    
    # === TRAINING ===
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # === OPTIONS ===
    FREEZE_ENCODERS = True   # Freeze pre-trained encoders (recommended)
    USE_AMP = True
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def print_config(cls):
        print(f"\n{'='*70}")
        print("⚙️  CONFIGURATION - FUSION FROM PRE-TRAINED")
        print(f"{'='*70}")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Best Vision:   {cls.BEST_VISION_MODEL}")
        print(f"  Best Signal:   {cls.BEST_SIGNAL_MODEL}")
        print(f"  Best Clinical: {cls.BEST_CLINICAL_MODEL}")
        print(f"  Freeze Encoders: {cls.FREEZE_ENCODERS}")
        print(f"{'='*70}\n")


# Check if checkpoints exist
def check_checkpoints():
    """Verify all checkpoints exist in Drive."""
    print("🔍 Checking for pre-trained model checkpoints...")
    
    checkpoints = {
        'Vision': Config.VISION_CHECKPOINT,
        'Signal': Config.SIGNAL_CHECKPOINT,
        'Clinical': Config.CLINICAL_CHECKPOINT
    }
    
    all_found = True
    for name, path in checkpoints.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✅ {name}: Found ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {name}: NOT FOUND at {path}")
            all_found = False
    
    if not all_found:
        print("\n⚠️ Some checkpoints missing! Please check the paths.")
        print("   Available files in checkpoint dir:")
        if os.path.exists(Config.CHECKPOINT_DIR):
            for f in os.listdir(Config.CHECKPOINT_DIR):
                print(f"     - {f}")
    
    return all_found


# ===================== CELL 4: DATASET =====================

class SymileMIMICMultiLabelDataset(Dataset):
    """Multi-Modal Dataset for SYMILE-MIMIC."""
    
    def __init__(self, data_dir, split='train'):
        self.split = split
        self.labels_list = Config.LABELS
        
        csv_path = f"{data_dir}/{split}.csv"
        npy_dir = f"{data_dir}/data_npy/{split}"
        
        print(f"  📂 Loading {split} data...")
        self.df = pd.read_csv(csv_path)
        
        # Load numpy arrays
        self.cxr = np.load(f"{npy_dir}/cxr_{split}.npy", mmap_mode='r')
        self.ecg = np.load(f"{npy_dir}/ecg_{split}.npy", mmap_mode='r')
        self.labs_percentiles = np.load(f"{npy_dir}/labs_percentiles_{split}.npy", mmap_mode='r')
        self.labs_missingness = np.load(f"{npy_dir}/labs_missingness_{split}.npy", mmap_mode='r')
        
        # Extract labels (handle uncertain = -1 as positive)
        self.labels = np.zeros((len(self.df), Config.NUM_LABELS), dtype=np.float32)
        for i, name in enumerate(self.labels_list):
            if name in self.df.columns:
                values = self.df[name].fillna(0).values
                self.labels[:, i] = ((values == 1.0) | (values == -1.0)).astype(float)
        
        print(f"  ✅ Loaded {len(self):,} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cxr = torch.from_numpy(self.cxr[idx].copy()).float()
        ecg = torch.from_numpy(self.ecg[idx].copy()).float()
        if ecg.dim() == 3:
            ecg = ecg.squeeze(0)
        if ecg.shape[0] != 12:
            ecg = ecg.transpose(0, 1)
        labs = torch.from_numpy(
            np.concatenate([self.labs_percentiles[idx], self.labs_missingness[idx]])
        ).float()
        labels = torch.from_numpy(self.labels[idx]).float()
        return cxr, ecg, labs, labels


# ===================== CELL 5: MODEL DEFINITIONS =====================

# --- Vision Models ---
class ConvNeXtTinyMultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "ConvNeXt-Tiny"
        self.feature_dim = 768
        self.backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Flatten(1), nn.LayerNorm(768), nn.Dropout(0.3),
            nn.Linear(768, num_labels)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.mean(dim=[-2, -1])
        logits = self.classifier(features)
        return logits, features


class DenseNet121MultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "DenseNet-121"
        self.feature_dim = 1024
        self.backbone = models.densenet121(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, num_labels))
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


class EfficientNetB2MultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "EfficientNet-B2"
        self.feature_dim = 1408
        self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1408, num_labels))
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


# --- Signal Models ---
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.Conv1d(out_ch, out_ch, 5, padding=2), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.branch4 = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.conv = nn.Conv1d(out_ch * 4, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
    
    def forward(self, x):
        b1, b2, b3, b4 = self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return F.relu(self.bn(self.conv(out)))


class InceptionTimeMultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "InceptionTime"
        self.feature_dim = 256
        self.stem = nn.Sequential(nn.Conv1d(12, 32, 1), nn.BatchNorm1d(32), nn.ReLU())
        self.blocks = nn.Sequential(
            InceptionBlock(32, 32), nn.MaxPool1d(2),
            InceptionBlock(32, 64), nn.MaxPool1d(2),
            InceptionBlock(64, 128), nn.MaxPool1d(2),
            InceptionBlock(128, 256), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_labels))
    
    def forward(self, x):
        x = self.stem(x)
        features = self.blocks(x).squeeze(-1)
        logits = self.classifier(features)
        return logits, features


class CNN1DMultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "1D-CNN"
        self.feature_dim = 256
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 15, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 11, padding=5), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 7, padding=3), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_labels))
    
    def forward(self, x):
        features = self.conv(x).squeeze(-1)
        logits = self.classifier(features)
        return logits, features


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch))
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1DMultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "ResNet-1D"
        self.feature_dim = 256
        self.stem = nn.Sequential(nn.Conv1d(12, 64, 15, stride=2, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(ResBlock1D(64, 64), ResBlock1D(64, 64))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128, stride=2), ResBlock1D(128, 128))
        self.layer3 = nn.Sequential(ResBlock1D(128, 256, stride=2), ResBlock1D(256, 256))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_labels))
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.avgpool(x).squeeze(-1)
        logits = self.classifier(features)
        return logits, features


# --- Clinical Models ---
class MLPMultiLabel(nn.Module):
    def __init__(self, input_dim=100, num_labels=6):
        super().__init__()
        self.name = "MLP"
        self.feature_dim = 64
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_labels))
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features


class TabNetMultiLabel(nn.Module):
    def __init__(self, input_dim=100, n_steps=3, hidden_dim=64, num_labels=6):
        super().__init__()
        self.name = "TabNet"
        self.feature_dim = 64
        self.n_steps = n_steps
        self.bn = nn.BatchNorm1d(input_dim)
        self.initial_fc = nn.Linear(input_dim, hidden_dim)
        self.attention_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.BatchNorm1d(input_dim), nn.Softmax(dim=-1))
            for _ in range(n_steps)
        ])
        self.fc_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
            for _ in range(n_steps)
        ])
        self.final_fc = nn.Linear(hidden_dim, self.feature_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.feature_dim, num_labels))
    
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


# --- Fusion Model ---
class MultiModalFusionMultiLabel(nn.Module):
    def __init__(self, vision_model, signal_model, clinical_model, num_labels=6):
        super().__init__()
        self.name = "MultiModal-Fusion"
        self.vision = vision_model
        self.signal = signal_model
        self.clinical = clinical_model
        
        total_features = vision_model.feature_dim + signal_model.feature_dim + clinical_model.feature_dim
        self.feature_dim = total_features
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )
        
        print(f"\n  🔀 Fusion Network: {vision_model.feature_dim} + {signal_model.feature_dim} + {clinical_model.feature_dim} = {total_features}D → {num_labels}")
    
    def forward(self, cxr, ecg, labs):
        _, vision_feat = self.vision(cxr)
        _, signal_feat = self.signal(ecg)
        _, clinical_feat = self.clinical(labs)
        combined = torch.cat([vision_feat, signal_feat, clinical_feat], dim=1)
        logits = self.fusion(combined)
        return logits, (vision_feat, signal_feat, clinical_feat)


# ===================== CELL 6: LOAD PRE-TRAINED MODELS =====================

def load_pretrained_model(model_class, checkpoint_path, model_name):
    """Load a pre-trained model from checkpoint."""
    print(f"  📥 Loading {model_name}...")
    
    model = model_class()
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=Config.DEVICE)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get metrics from checkpoint
    best_auc = checkpoint.get('best_auc', checkpoint.get('metrics', {}).get('macro', {}).get('auc', 0))
    epoch = checkpoint.get('epoch', 'N/A')
    
    print(f"     ✅ Loaded! Best AUC: {best_auc:.4f} (Epoch {epoch})")
    
    return model, best_auc


def get_model_class(modality, model_name):
    """Get the model class based on modality and name."""
    if modality == 'vision':
        classes = {'ConvNeXt-Tiny': ConvNeXtTinyMultiLabel, 'DenseNet-121': DenseNet121MultiLabel, 'EfficientNet-B2': EfficientNetB2MultiLabel}
    elif modality == 'signal':
        classes = {'InceptionTime': InceptionTimeMultiLabel, '1D-CNN': CNN1DMultiLabel, 'ResNet-1D': ResNet1DMultiLabel}
    else:
        classes = {'MLP': MLPMultiLabel, 'TabNet': TabNetMultiLabel}
    return classes.get(model_name)


# ===================== CELL 7: COMPREHENSIVE METRICS =====================

def compute_comprehensive_metrics(labels, probs, threshold=0.5):
    """
    Compute ALL metrics including Accuracy, Precision, Recall, F1, Specificity.
    """
    preds = (probs > threshold).astype(int)
    n_classes = labels.shape[1]
    
    results = {
        'per_class': {},
        'macro': {},
        'micro': {},
        'labels': labels,
        'probs': probs,
        'preds': preds
    }
    
    # Per-class metrics
    per_class_metrics = {k: [] for k in ['auc', 'ap', 'f1', 'precision', 'recall', 'specificity', 'accuracy']}
    
    for i in range(n_classes):
        class_name = Config.LABELS[i]
        y_true = labels[:, i]
        y_prob = probs[:, i]
        y_pred = preds[:, i]
        
        # Confusion matrix elements
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        
        # Calculate metrics
        try:
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) >= 2 else 0.5
        except:
            auc = 0.5
        
        try:
            ap = average_precision_score(y_true, y_prob)
        except:
            ap = 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        results['per_class'][class_name] = {
            'auc': auc, 'ap': ap, 'f1': f1,
            'precision': precision, 'recall': recall,
            'specificity': specificity, 'accuracy': accuracy,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
        
        per_class_metrics['auc'].append(auc)
        per_class_metrics['ap'].append(ap)
        per_class_metrics['f1'].append(f1)
        per_class_metrics['precision'].append(precision)
        per_class_metrics['recall'].append(recall)
        per_class_metrics['specificity'].append(specificity)
        per_class_metrics['accuracy'].append(accuracy)
    
    # Macro averages
    results['macro'] = {k: np.mean(v) for k, v in per_class_metrics.items()}
    
    # Micro averages (aggregate then compute)
    results['micro'] = {
        'auc': roc_auc_score(labels.ravel(), probs.ravel()),
        'f1': f1_score(labels.ravel(), preds.ravel()),
        'precision': precision_score(labels.ravel(), preds.ravel()),
        'recall': recall_score(labels.ravel(), preds.ravel()),
        'accuracy': accuracy_score(labels.ravel(), preds.ravel())
    }
    
    return results


def print_detailed_metrics(metrics, model_name):
    """Print a detailed metrics table."""
    print(f"\n{'='*90}")
    print(f"📊 {model_name}: COMPREHENSIVE METRICS")
    print(f"{'='*90}")
    
    # Header
    header = f"{'Disease':<20} | {'AUC':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'Spec':>7} | {'Acc':>7}"
    print(header)
    print("-" * 90)
    
    # Per-class metrics
    for label in Config.LABELS:
        m = metrics['per_class'][label]
        row = f"{label:<20} | {m['auc']:>7.4f} | {m['precision']:>7.4f} | {m['recall']:>7.4f} | {m['f1']:>7.4f} | {m['specificity']:>7.4f} | {m['accuracy']:>7.4f}"
        print(row)
    
    print("-" * 90)
    
    # Macro averages
    m = metrics['macro']
    row = f"{'MACRO AVERAGE':<20} | {m['auc']:>7.4f} | {m['precision']:>7.4f} | {m['recall']:>7.4f} | {m['f1']:>7.4f} | {m['specificity']:>7.4f} | {m['accuracy']:>7.4f}"
    print(row)
    
    # Micro averages
    m = metrics['micro']
    row = f"{'MICRO AVERAGE':<20} | {m['auc']:>7.4f} | {m['precision']:>7.4f} | {m['recall']:>7.4f} | {m['f1']:>7.4f} | {'N/A':>7} | {m['accuracy']:>7.4f}"
    print(row)
    
    print(f"{'='*90}\n")


# ===================== CELL 8: TRAINER =====================

class FusionTrainer:
    """Trainer for fusion model with frozen encoders."""
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Freeze encoder weights if configured
        if Config.FREEZE_ENCODERS:
            print("  ❄️ Freezing encoder weights...")
            for param in model.vision.parameters():
                param.requires_grad = False
            for param in model.signal.parameters():
                param.requires_grad = False
            for param in model.clinical.parameters():
                param.requires_grad = False
            
            # Only train fusion layers
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"     Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")
        
        # Optimizer only for trainable params
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=Config.EPOCHS)
        self.scaler = GradScaler() if Config.USE_AMP else None
        self.history = defaultdict(list)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for cxr, ecg, labs, labels in tqdm(self.train_loader, desc="Training", leave=False):
            cxr = cxr.to(Config.DEVICE)
            ecg = ecg.to(Config.DEVICE)
            labs = labs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=Config.USE_AMP):
                logits, _ = self.model(cxr, ecg, labs)
                # Label smoothing
                smooth_labels = labels * 0.9 + 0.05
                loss = F.binary_cross_entropy_with_logits(logits, smooth_labels)
            
            if Config.USE_AMP:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_probs, all_labels = [], []
        
        for cxr, ecg, labs, labels in tqdm(self.val_loader, desc="Evaluating", leave=False):
            cxr = cxr.to(Config.DEVICE)
            ecg = ecg.to(Config.DEVICE)
            labs = labs.to(Config.DEVICE)
            
            with autocast(enabled=Config.USE_AMP):
                logits, _ = self.model(cxr, ecg, labs)
            
            probs = torch.sigmoid(logits.float())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        return compute_comprehensive_metrics(all_labels, all_probs)
    
    def train(self, epochs):
        print(f"\n{'='*70}")
        print(f"🚀 Training: Multi-Modal Fusion")
        print(f"   Epochs: {epochs}")
        print(f"{'='*70}")
        
        best_auc = 0
        best_epoch = 0
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            train_loss = self.train_epoch()
            metrics = self.evaluate()
            
            self.scheduler.step()
            
            macro_auc = metrics['macro']['auc']
            self.history['train_loss'].append(train_loss)
            self.history['val_macro_auc'].append(macro_auc)
            
            marker = ""
            if macro_auc > best_auc:
                best_auc = macro_auc
                best_epoch = epoch + 1
                best_metrics = metrics
                patience_counter = 0
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'best_auc': best_auc,
                    'metrics': {k: v for k, v in metrics.items() if k not in ['labels', 'probs', 'preds']}
                }, f"{Config.CHECKPOINT_DIR}/fusion_best.pth")
                marker = " ✅ Best! (saved)"
            else:
                patience_counter += 1
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Macro-AUC: {macro_auc:.4f} | F1: {metrics['macro']['f1']:.4f} | "
                  f"Time: {epoch_time:.1f}s{marker}")
            
            if patience_counter >= 5:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Total training time: {total_time:.1f}s")
        print(f"🏆 Best Macro-AUC: {best_auc:.4f} (Epoch {best_epoch})")
        
        # Load best and evaluate
        checkpoint = torch.load(f"{Config.CHECKPOINT_DIR}/fusion_best.pth", weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        final_metrics = self.evaluate()
        
        return {
            'model_name': 'MultiModal-Fusion',
            'best_auc': best_auc,
            'best_epoch': best_epoch,
            'history': dict(self.history),
            'final_metrics': final_metrics,
            'train_time': total_time
        }


# ===================== CELL 9: VISUALIZATION FUNCTIONS =====================

def plot_fusion_architecture_diagram(output_dir):
    """
    Create a detailed architecture diagram of the Multi-Modal Fusion Model.
    Perfect for teacher presentation!
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    
    # Title
    ax.text(9, 13.5, '🔀 Multi-Modal Fusion Architecture for Disease Detection', 
            fontsize=20, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(9, 12.9, 'Intermediate Feature-Level Fusion | SYMILE-MIMIC Dataset', 
            fontsize=12, ha='center', color='#7f8c8d', style='italic')
    
    # ===== INPUT MODALITIES =====
    # Vision Input
    vision_box = plt.Rectangle((0.5, 9), 3.5, 3), 
    ax.add_patch(plt.Rectangle((0.5, 9), 3.5, 3, facecolor='#3498db', edgecolor='#2980b9', linewidth=3, alpha=0.9))
    ax.text(2.25, 11.5, '🩻 CHEST X-RAY', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(2.25, 10.8, 'Input: 224×224×3', fontsize=10, ha='center', color='white')
    ax.text(2.25, 10.2, 'RGB Image', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(2.25, 9.5, '(Radiograph)', fontsize=9, ha='center', color='#bdc3c7')
    
    # Signal Input
    ax.add_patch(plt.Rectangle((7.25, 9), 3.5, 3, facecolor='#e74c3c', edgecolor='#c0392b', linewidth=3, alpha=0.9))
    ax.text(9, 11.5, '❤️ 12-LEAD ECG', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(9, 10.8, 'Input: 12×5000', fontsize=10, ha='center', color='white')
    ax.text(9, 10.2, 'Time Series', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(9, 9.5, '(10 seconds)', fontsize=9, ha='center', color='#bdc3c7')
    
    # Clinical Input
    ax.add_patch(plt.Rectangle((14, 9), 3.5, 3, facecolor='#9b59b6', edgecolor='#8e44ad', linewidth=3, alpha=0.9))
    ax.text(15.75, 11.5, '🧪 LAB VALUES', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(15.75, 10.8, 'Input: 100D', fontsize=10, ha='center', color='white')
    ax.text(15.75, 10.2, '50 Values + 50 Flags', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(15.75, 9.5, '(Blood Tests)', fontsize=9, ha='center', color='#bdc3c7')
    
    # ===== ARROWS: Input to Encoder =====
    ax.annotate('', xy=(2.25, 7.5), xytext=(2.25, 9), 
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=3))
    ax.annotate('', xy=(9, 7.5), xytext=(9, 9), 
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))
    ax.annotate('', xy=(15.75, 7.5), xytext=(15.75, 9), 
                arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=3))
    
    # ===== ENCODERS (Pre-trained) =====
    # Vision Encoder
    ax.add_patch(plt.Rectangle((0.5, 5.5), 3.5, 2, facecolor='#2980b9', edgecolor='#1a5276', linewidth=2))
    ax.text(2.25, 7, 'ConvNeXt-Tiny', fontsize=11, fontweight='bold', ha='center', color='white')
    ax.text(2.25, 6.4, '(Pre-trained)', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(2.25, 5.9, '❄️ Frozen', fontsize=8, ha='center', color='#f39c12')
    
    # Signal Encoder
    ax.add_patch(plt.Rectangle((7.25, 5.5), 3.5, 2, facecolor='#c0392b', edgecolor='#922b21', linewidth=2))
    ax.text(9, 7, '1D-CNN', fontsize=11, fontweight='bold', ha='center', color='white')
    ax.text(9, 6.4, '(Pre-trained)', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(9, 5.9, '❄️ Frozen', fontsize=8, ha='center', color='#f39c12')
    
    # Clinical Encoder
    ax.add_patch(plt.Rectangle((14, 5.5), 3.5, 2, facecolor='#8e44ad', edgecolor='#6c3483', linewidth=2))
    ax.text(15.75, 7, 'MLP', fontsize=11, fontweight='bold', ha='center', color='white')
    ax.text(15.75, 6.4, '(Pre-trained)', fontsize=9, ha='center', color='#bdc3c7')
    ax.text(15.75, 5.9, '❄️ Frozen', fontsize=8, ha='center', color='#f39c12')
    
    # ===== FEATURE VECTORS =====
    ax.annotate('', xy=(2.25, 4.5), xytext=(2.25, 5.5), 
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2))
    ax.annotate('', xy=(9, 4.5), xytext=(9, 5.5), 
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))
    ax.annotate('', xy=(15.75, 4.5), xytext=(15.75, 5.5), 
                arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2))
    
    # Feature dimension boxes
    ax.add_patch(plt.Rectangle((1, 4), 2.5, 0.5, facecolor='#5dade2', edgecolor='#2980b9', linewidth=1))
    ax.text(2.25, 4.25, '768D', fontsize=10, fontweight='bold', ha='center', color='white')
    
    ax.add_patch(plt.Rectangle((7.75, 4), 2.5, 0.5, facecolor='#ec7063', edgecolor='#c0392b', linewidth=1))
    ax.text(9, 4.25, '256D', fontsize=10, fontweight='bold', ha='center', color='white')
    
    ax.add_patch(plt.Rectangle((14.5, 4), 2.5, 0.5, facecolor='#bb8fce', edgecolor='#8e44ad', linewidth=1))
    ax.text(15.75, 4.25, '64D', fontsize=10, fontweight='bold', ha='center', color='white')
    
    # ===== CONCATENATION =====
    # Arrows converging to center
    ax.annotate('', xy=(7, 3.25), xytext=(2.25, 4), 
                arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    ax.annotate('', xy=(9, 3.25), xytext=(9, 4), 
                arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    ax.annotate('', xy=(11, 3.25), xytext=(15.75, 4), 
                arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
    
    # Concatenation box
    ax.add_patch(plt.Rectangle((6, 2.5), 6, 0.75, facecolor='#1abc9c', edgecolor='#16a085', linewidth=2))
    ax.text(9, 2.9, '🔗 CONCATENATE: 768 + 256 + 64 = 1088D', fontsize=11, fontweight='bold', ha='center', color='white')
    
    # ===== FUSION MLP =====
    ax.annotate('', xy=(9, 1.75), xytext=(9, 2.5), 
                arrowprops=dict(arrowstyle='->', color='#1abc9c', lw=3))
    
    ax.add_patch(plt.Rectangle((5.5, 0.75), 7, 1, facecolor='#f39c12', edgecolor='#d68910', linewidth=3))
    ax.text(9, 1.5, '🧠 FUSION MLP (Trainable)', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(9, 1.0, '1088 → 512 → 256 → 128 → 6', fontsize=10, ha='center', color='white')
    
    # ===== OUTPUT =====
    ax.annotate('', xy=(9, 0), xytext=(9, 0.75), 
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3))
    
    # Output predictions (6 diseases)
    diseases = ['Cardiomegaly', 'Edema', 'Atelectasis', 'Pleural\nEffusion', 'Lung\nOpacity', 'No\nFinding']
    x_positions = [3, 5.4, 7.8, 10.2, 12.6, 15]
    
    for i, (x, disease) in enumerate(zip(x_positions, diseases)):
        ax.add_patch(plt.Rectangle((x-1, -1.2), 2, 0.9, facecolor='#27ae60', edgecolor='#1e8449', linewidth=1.5, alpha=0.9))
        ax.text(x, -0.7, disease, fontsize=8, fontweight='bold', ha='center', va='center', color='white')
    
    ax.text(9, -1.7, '📊 6 Disease Predictions (Multi-Label Sigmoid Output)', 
            fontsize=11, ha='center', color='#2c3e50', style='italic')
    
    # ===== LEGEND BOX =====
    legend_box = plt.Rectangle((13.5, -2.5), 4, 2.2, facecolor='white', edgecolor='#bdc3c7', linewidth=1)
    ax.add_patch(legend_box)
    ax.text(15.5, -0.5, 'LEGEND', fontsize=9, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(13.8, -0.9, '❄️ = Frozen (Pre-trained)', fontsize=8, ha='left', color='#7f8c8d')
    ax.text(13.8, -1.3, '🔥 = Trainable', fontsize=8, ha='left', color='#7f8c8d')
    ax.text(13.8, -1.7, '→ = Data Flow', fontsize=8, ha='left', color='#7f8c8d')
    ax.text(13.8, -2.1, 'D = Dimension', fontsize=8, ha='left', color='#7f8c8d')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fusion_architecture_diagram.png", dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  ✅ Saved: fusion_architecture_diagram.png")
    plt.close()


def plot_confusion_matrices(metrics, output_dir):
    """Plot confusion matrices for each disease class."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, label in enumerate(Config.LABELS):
        m = metrics['per_class'][label]
        cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'],
                    annot_kws={'size': 14})
        axes[i].set_title(f'{label}\nAUC: {m["auc"]:.3f} | F1: {m["f1"]:.3f}', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')
    
    plt.suptitle('📊 Confusion Matrices - Multi-Modal Fusion', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: confusion_matrices.png")
    plt.close()


def plot_roc_curves(metrics, output_dir):
    """Plot ROC curves for all disease classes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(Config.LABELS)))
    
    for i, (label, color) in enumerate(zip(Config.LABELS, colors)):
        y_true = metrics['labels'][:, i]
        y_prob = metrics['probs'][:, i]
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = metrics['per_class'][label]['auc']
        
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={auc_val:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('📈 ROC Curves - Multi-Modal Fusion', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: roc_curves.png")
    plt.close()


def plot_per_class_metrics(metrics, output_dir):
    """Bar chart comparing metrics across diseases."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    labels = Config.LABELS
    x = np.arange(len(labels))
    width = 0.6
    
    metrics_to_plot = [
        ('auc', 'AUC-ROC', '#3498db'),
        ('f1', 'F1-Score', '#e74c3c'),
        ('precision', 'Precision', '#2ecc71'),
        ('recall', 'Recall/Sensitivity', '#9b59b6')
    ]
    
    for ax, (metric_key, metric_name, color) in zip(axes.flatten(), metrics_to_plot):
        values = [metrics['per_class'][l][metric_key] for l in labels]
        macro_avg = metrics['macro'][metric_key]
        
        bars = ax.bar(x, values, width, color=color, edgecolor='black', alpha=0.8)
        ax.axhline(y=macro_avg, color='red', linestyle='--', lw=2, label=f'Macro Avg: {macro_avg:.3f}')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right')
        ax.set_title(f'{metric_name} per Disease', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('📊 Per-Disease Performance Metrics - Multi-Modal Fusion', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_class_metrics.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: per_class_metrics.png")
    plt.close()


def plot_training_history(history, output_dir):
    """Plot training loss and validation AUC over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', lw=2, markersize=6, label='Training Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('📉 Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # AUC
    axes[1].plot(epochs, history['val_macro_auc'], 'g-o', lw=2, markersize=6, label='Validation Macro-AUC')
    best_epoch = np.argmax(history['val_macro_auc']) + 1
    best_auc = max(history['val_macro_auc'])
    axes[1].axvline(x=best_epoch, color='red', linestyle='--', lw=1.5, label=f'Best Epoch: {best_epoch}')
    axes[1].scatter([best_epoch], [best_auc], color='red', s=100, zorder=5, marker='*')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Macro-AUC', fontsize=12)
    axes[1].set_title(f'📈 Validation AUC (Best: {best_auc:.4f})', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.suptitle('🔀 Multi-Modal Fusion Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: training_history.png")
    plt.close()


def plot_modality_comparison(vision_auc, signal_auc, clinical_auc, fusion_auc, output_dir):
    """Bar chart comparing individual modalities vs fusion."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    modalities = ['Vision\n(ConvNeXt-Tiny)', 'Signal\n(1D-CNN)', 'Clinical\n(MLP)', '🔀 FUSION']
    aucs = [vision_auc, signal_auc, clinical_auc, fusion_auc]
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#27ae60']
    
    bars = ax.bar(modalities, aucs, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auc:.4f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Highlight fusion improvement
    best_single = max(vision_auc, signal_auc, clinical_auc)
    ax.axhline(y=best_single, color='gray', linestyle='--', lw=1.5, label=f'Best Single: {best_single:.4f}')
    
    improvement = fusion_auc - best_single
    if improvement > 0:
        ax.annotate(f'+{improvement:.4f}', xy=(3, fusion_auc), xytext=(3.5, fusion_auc - 0.03),
                    fontsize=12, color='#27ae60', fontweight='bold')
    
    ax.set_ylabel('Macro-AUC', fontsize=12)
    ax.set_ylim(0.5, 0.85)
    ax.set_title('📊 Modality Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/modality_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: modality_comparison.png")
    plt.close()


def generate_all_visualizations(result, vision_auc, signal_auc, clinical_auc, output_dir):
    """Generate all visualizations."""
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS...")
    print("="*70)
    
    metrics = result['final_metrics']
    history = result['history']
    
    # 1. Architecture diagram (for teacher!)
    plot_fusion_architecture_diagram(output_dir)
    
    # 2. Confusion matrices
    plot_confusion_matrices(metrics, output_dir)
    
    # 3. ROC curves
    plot_roc_curves(metrics, output_dir)
    
    # 4. Per-class metrics bar chart
    plot_per_class_metrics(metrics, output_dir)
    
    # 5. Training history
    plot_training_history(history, output_dir)
    
    # 6. Modality comparison
    plot_modality_comparison(vision_auc, signal_auc, clinical_auc, result['best_auc'], output_dir)
    
    print("\n✅ All visualizations generated!")


# ===================== CELL 10: MAIN =====================

def main():
    """Main function to train fusion from pre-trained models."""
    
    print("\n" + "="*70)
    print("🔀 VISIONCARE: FUSION FROM PRE-TRAINED MODELS")
    print("="*70)
    
    Config.print_config()
    
    # Check checkpoints
    if not check_checkpoints():
        print("\n❌ Cannot proceed without all checkpoints!")
        return
    
    # Create output dirs
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Load data
    print("\n📂 Loading datasets...")
    train_ds = SymileMIMICMultiLabelDataset(Config.DATA_DIR, 'train')
    val_ds = SymileMIMICMultiLabelDataset(Config.DATA_DIR, 'val')
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Load pre-trained models
    print("\n📥 Loading pre-trained models from Google Drive...")
    
    vision_class = get_model_class('vision', Config.BEST_VISION_MODEL)
    signal_class = get_model_class('signal', Config.BEST_SIGNAL_MODEL)
    clinical_class = get_model_class('clinical', Config.BEST_CLINICAL_MODEL)
    
    vision_model, vision_auc = load_pretrained_model(vision_class, Config.VISION_CHECKPOINT, Config.BEST_VISION_MODEL)
    signal_model, signal_auc = load_pretrained_model(signal_class, Config.SIGNAL_CHECKPOINT, Config.BEST_SIGNAL_MODEL)
    clinical_model, clinical_auc = load_pretrained_model(clinical_class, Config.CLINICAL_CHECKPOINT, Config.BEST_CLINICAL_MODEL)
    
    print(f"\n  📊 Pre-trained Model Performance:")
    print(f"     Vision ({Config.BEST_VISION_MODEL}):    {vision_auc:.4f}")
    print(f"     Signal ({Config.BEST_SIGNAL_MODEL}):   {signal_auc:.4f}")
    print(f"     Clinical ({Config.BEST_CLINICAL_MODEL}): {clinical_auc:.4f}")
    
    # Create fusion model
    fusion_model = MultiModalFusionMultiLabel(vision_model, signal_model, clinical_model)
    
    # Train fusion
    trainer = FusionTrainer(fusion_model, train_loader, val_loader)
    result = trainer.train(Config.EPOCHS)
    
    # Print comprehensive metrics
    print_detailed_metrics(result['final_metrics'], 'Multi-Modal Fusion')
    
    # ========================================
    # 📊 GENERATE ALL VISUALIZATIONS
    # ========================================
    generate_all_visualizations(result, vision_auc, signal_auc, clinical_auc, Config.OUTPUT_DIR)
    
    # Save detailed results
    metrics_summary = {
        'model': 'MultiModal-Fusion',
        'components': {
            'vision': {'name': Config.BEST_VISION_MODEL, 'auc': vision_auc},
            'signal': {'name': Config.BEST_SIGNAL_MODEL, 'auc': signal_auc},
            'clinical': {'name': Config.BEST_CLINICAL_MODEL, 'auc': clinical_auc}
        },
        'fusion': {
            'best_auc': result['best_auc'],
            'best_epoch': result['best_epoch'],
            'train_time_seconds': result['train_time']
        },
        'macro_metrics': result['final_metrics']['macro'],
        'micro_metrics': result['final_metrics']['micro'],
        'per_class': {k: {kk: vv for kk, vv in v.items() if kk not in ['tp', 'tn', 'fp', 'fn']} 
                      for k, v in result['final_metrics']['per_class'].items()}
    }
    
    with open(f"{Config.OUTPUT_DIR}/fusion_metrics_detailed.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 FINAL MODEL COMPARISON:")
    print(f"  {'Modality':<12} | {'Model':<18} | {'Macro-AUC':>10}")
    print(f"  {'-'*12}-+-{'-'*18}-+-{'-'*10}")
    print(f"  {'Vision':<12} | {Config.BEST_VISION_MODEL:<18} | {vision_auc:>10.4f}")
    print(f"  {'Signal':<12} | {Config.BEST_SIGNAL_MODEL:<18} | {signal_auc:>10.4f}")
    print(f"  {'Clinical':<12} | {Config.BEST_CLINICAL_MODEL:<18} | {clinical_auc:>10.4f}")
    print(f"  {'-'*12}-+-{'-'*18}-+-{'-'*10}")
    print(f"  {'🔀 FUSION':<12} | {'Multi-Modal':<18} | {result['best_auc']:>10.4f} 🏆")
    
    improvement = result['best_auc'] - max(vision_auc, signal_auc, clinical_auc)
    print(f"\n  📈 Fusion Improvement: {improvement:+.4f} over best single modality")
    
    print(f"\n📁 Files saved to: {Config.OUTPUT_DIR}/")
    print("   • fusion_architecture_diagram.png  ← For your teacher! 📐")
    print("   • confusion_matrices.png")
    print("   • roc_curves.png")
    print("   • per_class_metrics.png")
    print("   • training_history.png")
    print("   • modality_comparison.png")
    print("   • fusion_metrics_detailed.json")
    
    print("\n✨ MULTI-MODAL FUSION SUCCESS! ✨")
    
    return result


# ===================== RUN =====================

if __name__ == "__main__":
    main()

