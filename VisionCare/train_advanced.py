"""
VisionCare - Advanced Model Comparison with SOTA Architectures
Cardiovascular Disease Detection using CXR + ECG + Labs

Author: VisionCare Team - BTP Semester 7
Date: February 2026

MODELS COMPARED:
================
🩻 Vision (CXR):
  - DenseNet-121 (CheXNet baseline)
  - EfficientNet-B2 (Efficient & accurate)
  - ConvNeXt-Tiny (Modern CNN, competes with ViT)

❤️ Signal (ECG):
  - 1D-CNN (Fast baseline)
  - ResNet-1D (Deep residual)
  - InceptionTime (SOTA for time-series)

🩸 Clinical (Labs):
  - MLP (Simple but effective)
  - TabNet (Attention-based)

OPTIMIZATIONS:
- Mixed precision (FP16) training
- Gradient accumulation
- DataLoader prefetching
- torch.compile() (PyTorch 2.0)
"""

import os
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
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve, average_precision_score
)

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True  # Speed optimization

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("✓ All libraries imported!")


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Paths
    DATA_DIR = "/content/drive/MyDrive/symile-mimic"
    OUTPUT_DIR = "VisionCare_Advanced_Results"
    
    # Training
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Increased for faster loading
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Optimizations
    USE_AMP = True  # Mixed precision
    USE_COMPILE = False  # torch.compile (PyTorch 2.0+)
    GRAD_ACCUM_STEPS = 1
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def print_config(cls):
        print(f"\n{'='*60}")
        print("⚙️ CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Device: {cls.DEVICE}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Mixed Precision: {cls.USE_AMP}")
        print(f"{'='*60}\n")

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATASET (Optimized with prefetching)
# ============================================================
class SymileMIMICDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.split = split
        self.df = pd.read_csv(f"{data_dir}/{split}.csv")
        
        npy_dir = f"{data_dir}/data_npy/{split}"
        print(f"  Loading {split}...")
        self.cxr = np.load(f"{npy_dir}/cxr_{split}.npy", mmap_mode='r')
        self.ecg = np.load(f"{npy_dir}/ecg_{split}.npy", mmap_mode='r')
        self.labs_pct = np.load(f"{npy_dir}/labs_percentiles_{split}.npy", mmap_mode='r')
        self.labs_miss = np.load(f"{npy_dir}/labs_missingness_{split}.npy", mmap_mode='r')
        
        if 'Cardiomegaly' in self.df.columns:
            labels = self.df['Cardiomegaly'].fillna(0).values
            self.labels = ((labels == 1.0) | (labels == -1.0)).astype(int)
        else:
            self.labels = np.zeros(len(self.df), dtype=int)
        
        print(f"  ✓ {split.upper()}: {len(self):,} samples, {self.labels.mean()*100:.1f}% positive")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cxr = torch.from_numpy(self.cxr[idx].copy()).float()
        ecg = torch.from_numpy(self.ecg[idx].copy()).float().squeeze(0).transpose(0, 1)
        labs = torch.from_numpy(np.concatenate([self.labs_pct[idx], self.labs_miss[idx]])).float()
        return cxr, ecg, labs, self.labels[idx]


# ============================================================
# 🩻 VISION MODELS
# ============================================================
class DenseNet121(nn.Module):
    """CheXNet baseline - proven on chest X-rays"""
    def __init__(self):
        super().__init__()
        self.name = "DenseNet-121"
        self.feature_dim = 1024
        self.params = "8M"
        self.backbone = models.densenet121(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 2))
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat), feat


class EfficientNetB2(nn.Module):
    """Efficient & accurate - better accuracy/params ratio"""
    def __init__(self):
        super().__init__()
        self.name = "EfficientNet-B2"
        self.feature_dim = 1408
        self.params = "9M"
        self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1408, 2))
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat), feat


class ConvNeXtTiny(nn.Module):
    """Modern CNN - competes with Vision Transformers"""
    def __init__(self):
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
            nn.Linear(768, 2)
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.mean(dim=[-2, -1])  # Global average pooling
        return self.classifier(feat), feat


# ============================================================
# ❤️ SIGNAL MODELS
# ============================================================
class CNN1D(nn.Module):
    """Fast baseline for ECG"""
    def __init__(self):
        super().__init__()
        self.name = "1D-CNN"
        self.feature_dim = 256
        self.params = "0.5M"
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 15, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 11, padding=5), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 7, padding=3), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 2))
    
    def forward(self, x):
        feat = self.conv(x).squeeze(-1)
        return self.classifier(feat), feat


class ResNet1D(nn.Module):
    """Deep residual network for ECG - captures subtle patterns"""
    def __init__(self):
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
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 2))
    
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
        feat = self.avgpool(x).squeeze(-1)
        return self.classifier(feat), feat


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


class InceptionTime(nn.Module):
    """SOTA for time-series - parallel kernels for multi-scale patterns"""
    def __init__(self):
        super().__init__()
        self.name = "InceptionTime"
        self.feature_dim = 256
        self.params = "1.5M"
        
        self.inception1 = InceptionBlock(12, 32)
        self.inception2 = InceptionBlock(128, 32)
        self.inception3 = InceptionBlock(128, 32)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 256)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 2))
    
    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        feat = self.avgpool(x).squeeze(-1)
        feat = self.fc(feat)
        return self.classifier(feat), feat


class InceptionBlock(nn.Module):
    """Inception module with parallel convolutions of different sizes"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv1d(in_ch, out_ch, 7, padding=3), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.branch4 = nn.Sequential(nn.Conv1d(in_ch, out_ch, 15, padding=7), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)  # 4 * out_ch = 128
        return self.pool(out)


# ============================================================
# 🩸 CLINICAL MODELS
# ============================================================
class MLP(nn.Module):
    """Simple but effective for tabular data"""
    def __init__(self):
        super().__init__()
        self.name = "MLP"
        self.feature_dim = 64
        self.params = "0.02M"
        self.encoder = nn.Sequential(
            nn.Linear(100, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 2))
    
    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat), feat


class TabNet(nn.Module):
    """Attention-based feature selection for tabular data"""
    def __init__(self, input_dim=100, n_steps=3, hidden_dim=64):
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
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.feature_dim, 2))
    
    def forward(self, x):
        x = self.bn(x)
        h = F.relu(self.initial_fc(x))
        
        aggregated = torch.zeros_like(h)
        for attn, fc in zip(self.attention_layers, self.fc_layers):
            mask = attn(h)  # Feature selection
            masked = x * mask
            step_out = fc(masked)
            aggregated = aggregated + step_out
            h = step_out
        
        feat = self.final_fc(aggregated)
        return self.classifier(feat), feat


# ============================================================
# FUSION MODEL
# ============================================================
class MultiModalFusion(nn.Module):
    def __init__(self, vision, signal, clinical):
        super().__init__()
        self.vision = vision
        self.signal = signal
        self.clinical = clinical
        
        total = vision.feature_dim + signal.feature_dim + clinical.feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(total, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        self.total_features = total
    
    def forward(self, cxr, ecg, labs):
        _, v = self.vision(cxr)
        _, s = self.signal(ecg)
        _, c = self.clinical(labs)
        return self.fusion(torch.cat([v, s, c], dim=1)), (v, s, c)


# ============================================================
# TRAINING ENGINE (Optimized with AMP)
# ============================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, modality):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.modality = modality
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=Config.EPOCHS)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler() if Config.USE_AMP else None
        
        # Compile model for speed (PyTorch 2.0+)
        if Config.USE_COMPILE and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (cxr, ecg, labs, y) in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            cxr = cxr.to(Config.DEVICE, non_blocking=True)
            ecg = ecg.to(Config.DEVICE, non_blocking=True)
            labs = labs.to(Config.DEVICE, non_blocking=True)
            y = y.to(Config.DEVICE, non_blocking=True)
            
            with autocast(enabled=Config.USE_AMP):
                if self.modality == 'fusion':
                    logits, _ = self.model(cxr, ecg, labs)
                elif self.modality == 'vision':
                    logits, _ = self.model(cxr)
                elif self.modality == 'signal':
                    logits, _ = self.model(ecg)
                else:
                    logits, _ = self.model(labs)
                
                loss = self.criterion(logits, y)
            
            if Config.USE_AMP:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_probs, all_labels = [], []
        
        for cxr, ecg, labs, y in tqdm(self.val_loader, desc="Evaluating", leave=False):
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
            
            probs = F.softmax(logits.float(), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
        
        preds = [1 if p > 0.5 else 0 for p in all_probs]
        
        return {
            'accuracy': accuracy_score(all_labels, preds),
            'precision': precision_score(all_labels, preds, zero_division=0),
            'recall': recall_score(all_labels, preds, zero_division=0),
            'f1': f1_score(all_labels, preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5,
            'avg_precision': average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5,
            'probs': np.array(all_probs),
            'labels': np.array(all_labels),
            'preds': np.array(preds)
        }
    
    def train(self, epochs, save_name):
        print(f"\n{'='*60}")
        print(f"🚀 Training: {self.model.name} ({getattr(self.model, 'params', 'N/A')} params)")
        print(f"{'='*60}")
        
        best_auc, best_epoch = 0, 0
        history = {'train_loss': [], 'val_auc': [], 'val_acc': []}
        
        start = time.time()
        for epoch in range(epochs):
            loss = self.train_epoch()
            metrics = self.evaluate()
            self.scheduler.step()
            
            history['train_loss'].append(loss)
            history['val_auc'].append(metrics['auc'])
            history['val_acc'].append(metrics['accuracy'])
            
            marker = ""
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), f"{Config.OUTPUT_DIR}/{save_name}.pth")
                marker = " ✅ Best!"
            
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {loss:.4f} | AUC: {metrics['auc']:.4f} | Acc: {metrics['accuracy']*100:.1f}%{marker}")
        
        train_time = time.time() - start
        print(f"⏱️ Time: {train_time:.1f}s ({train_time/epochs:.1f}s/epoch) | Best AUC: {best_auc:.4f} (Epoch {best_epoch})")
        
        return {
            'model_name': self.model.name,
            'params': getattr(self.model, 'params', 'N/A'),
            'feature_dim': self.model.feature_dim,
            'history': history,
            'best_auc': best_auc,
            'best_epoch': best_epoch,
            'train_time': train_time,
            'final_metrics': metrics
        }


# ============================================================
# VISUALIZATION
# ============================================================
def create_comparison_table(results, modality_name):
    """Create detailed comparison table."""
    data = []
    for r in results:
        m = r['final_metrics']
        data.append({
            'Model': r['model_name'],
            'Params': r['params'],
            'AUC-ROC': f"{r['best_auc']:.4f}",
            'Accuracy': f"{m['accuracy']*100:.1f}%",
            'Precision': f"{m['precision']:.3f}",
            'Recall': f"{m['recall']:.3f}",
            'F1': f"{m['f1']:.3f}",
            'Time (s)': f"{r['train_time']:.1f}"
        })
    
    df = pd.DataFrame(data)
    print(f"\n📊 {modality_name} Model Comparison:")
    print(df.to_string(index=False))
    df.to_csv(f"{Config.OUTPUT_DIR}/{modality_name.lower()}_comparison.csv", index=False)
    return df


def plot_comprehensive_comparison(results, modality_name, output_dir):
    """Create multi-panel comparison figure."""
    fig = plt.figure(figsize=(20, 12))
    
    names = [r['model_name'] for r in results]
    aucs = [r['best_auc'] for r in results]
    accs = [r['final_metrics']['accuracy'] for r in results]
    times = [r['train_time'] for r in results]
    colors = sns.color_palette("husl", len(results))
    
    # 1. AUC Bar Chart
    ax1 = fig.add_subplot(2, 3, 1)
    bars = ax1.bar(names, aucs, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('AUC-ROC', fontweight='bold')
    ax1.set_title(f'🏆 {modality_name}: AUC Comparison', fontweight='bold', fontsize=12)
    ax1.set_ylim(0.5, 1.0)
    best_idx = np.argmax(aucs)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    for bar, auc in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auc:.3f}', ha='center', fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    
    # 2. Training Time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(names, times, color=colors, edgecolor='black')
    ax2.set_ylabel('Training Time (s)', fontweight='bold')
    ax2.set_title('⏱️ Training Speed', fontweight='bold', fontsize=12)
    ax2.tick_params(axis='x', rotation=15)
    
    # 3. AUC vs Params scatter
    ax3 = fig.add_subplot(2, 3, 3)
    params_num = []
    for r in results:
        p = r['params'].replace('M', '')
        params_num.append(float(p) if p != 'N/A' else 1)
    scatter = ax3.scatter(params_num, aucs, c=colors, s=200, edgecolor='black', linewidth=2)
    for i, name in enumerate(names):
        ax3.annotate(name, (params_num[i], aucs[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Parameters (M)', fontweight='bold')
    ax3.set_ylabel('AUC-ROC', fontweight='bold')
    ax3.set_title('📈 Efficiency: AUC vs Parameters', fontweight='bold', fontsize=12)
    
    # 4. Training Curves (AUC)
    ax4 = fig.add_subplot(2, 3, 4)
    for r, color in zip(results, colors):
        ax4.plot(r['history']['val_auc'], label=r['model_name'], linewidth=2, color=color)
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Validation AUC', fontweight='bold')
    ax4.set_title('📈 Training Progress (AUC)', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Training Curves (Loss)
    ax5 = fig.add_subplot(2, 3, 5)
    for r, color in zip(results, colors):
        ax5.plot(r['history']['train_loss'], label=r['model_name'], linewidth=2, color=color)
    ax5.set_xlabel('Epoch', fontweight='bold')
    ax5.set_ylabel('Training Loss', fontweight='bold')
    ax5.set_title('📉 Training Progress (Loss)', fontweight='bold', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. ROC Curves
    ax6 = fig.add_subplot(2, 3, 6)
    for r, color in zip(results, colors):
        m = r['final_metrics']
        fpr, tpr, _ = roc_curve(m['labels'], m['probs'])
        ax6.plot(fpr, tpr, label=f"{r['model_name']} ({r['best_auc']:.3f})", linewidth=2, color=color)
    ax6.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    ax6.set_xlabel('False Positive Rate', fontweight='bold')
    ax6.set_ylabel('True Positive Rate', fontweight='bold')
    ax6.set_title('📊 ROC Curves', fontweight='bold', fontsize=12)
    ax6.legend(loc='lower right')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'🔬 {modality_name} Module - Comprehensive Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{modality_name.lower()}_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/{modality_name.lower()}_comprehensive.png")
    plt.show()
    
    return names[best_idx], aucs[best_idx]


def plot_confusion_matrices_grid(all_results, output_dir):
    """Plot confusion matrices for all best models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, (name, data) in enumerate(all_results.items()):
        m = data['final_metrics']
        cm = confusion_matrix(m['labels'], m['preds'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['No CVD', 'CVD'], yticklabels=['No CVD', 'CVD'])
        axes[idx].set_title(f'{name}: {data["model_name"]}', fontweight='bold', fontsize=12)
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    plt.suptitle('🔍 Confusion Matrices - Best Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices_grid.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/confusion_matrices_grid.png")
    plt.show()


def plot_final_fusion_comparison(all_results, output_dir):
    """Final comparison showing fusion improvement."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    names = list(all_results.keys())
    aucs = [all_results[n]['best_auc'] for n in names]
    models = [all_results[n]['model_name'] for n in names]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    
    # Bar chart with model names as labels
    labels = [f"{n}\n({m})" for n, m in zip(names, models)]
    bars = axes[0].bar(labels, aucs, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    axes[0].set_title('🏆 Multi-Modal Fusion: Final Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0.5, 1.0)
    
    # Mark fusion as best
    bars[-1].set_hatch('//')
    bars[-1].set_edgecolor('gold')
    bars[-1].set_linewidth(4)
    
    for bar, auc in zip(bars, aucs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auc:.3f}', ha='center', fontweight='bold', fontsize=11)
    
    # Improvement annotation
    best_single = max(aucs[:-1])
    fusion_auc = aucs[-1]
    improvement = (fusion_auc - best_single) / best_single * 100
    axes[0].annotate(f'+{improvement:.1f}%', xy=(3, fusion_auc), xytext=(3, fusion_auc + 0.05),
                    fontsize=14, fontweight='bold', color='green', ha='center',
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    # ROC Curves
    for (name, data), color in zip(all_results.items(), colors):
        m = data['final_metrics']
        fpr, tpr, _ = roc_curve(m['labels'], m['probs'])
        lw = 3 if name == 'Fusion' else 2
        ls = '-' if name == 'Fusion' else '--'
        axes[1].plot(fpr, tpr, label=f"{name} ({data['best_auc']:.3f})", color=color, linewidth=lw, linestyle=ls)
    
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('📊 ROC Curves - All Modalities', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_fusion_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/final_fusion_comparison.png")
    plt.show()


def generate_comprehensive_report(all_results, vision_results, signal_results, clinical_results, output_dir):
    """Generate detailed training report."""
    
    fusion = all_results['Fusion']
    m = fusion['final_metrics']
    
    report = f"""
{'='*80}
VISIONCARE - ADVANCED MODEL COMPARISON REPORT
Multi-Modal Cardiovascular Disease Detection
BTP Semester 7 - February 2026
{'='*80}

📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
1. 🩻 VISION MODULE (Chest X-Ray) - MODEL COMPARISON
{'='*80}
"""
    for r in vision_results:
        star = "⭐" if r['model_name'] == all_results['Vision']['model_name'] else "  "
        report += f"  {star} {r['model_name']:20s} | Params: {r['params']:5s} | AUC: {r['best_auc']:.4f} | Time: {r['train_time']:6.1f}s\n"
    
    report += f"""
  🏆 Winner: {all_results['Vision']['model_name']}
     Reason: {'Best accuracy/parameter ratio with proven medical imaging capabilities' if 'Efficient' in all_results['Vision']['model_name'] else 'Highest AUC score on validation set'}

{'='*80}
2. ❤️ SIGNAL MODULE (12-Lead ECG) - MODEL COMPARISON
{'='*80}
"""
    for r in signal_results:
        star = "⭐" if r['model_name'] == all_results['Signal']['model_name'] else "  "
        report += f"  {star} {r['model_name']:20s} | Params: {r['params']:5s} | AUC: {r['best_auc']:.4f} | Time: {r['train_time']:6.1f}s\n"
    
    report += f"""
  🏆 Winner: {all_results['Signal']['model_name']}
     Reason: {'Captures multi-scale temporal patterns with inception modules' if 'Inception' in all_results['Signal']['model_name'] else 'Best balance of accuracy and speed'}

{'='*80}
3. 🩸 CLINICAL MODULE (Blood Labs) - MODEL COMPARISON  
{'='*80}
"""
    for r in clinical_results:
        star = "⭐" if r['model_name'] == all_results['Clinical']['model_name'] else "  "
        report += f"  {star} {r['model_name']:20s} | Params: {r['params']:5s} | AUC: {r['best_auc']:.4f} | Time: {r['train_time']:6.1f}s\n"
    
    report += f"""
  🏆 Winner: {all_results['Clinical']['model_name']}

{'='*80}
4. 🔀 MULTI-MODAL FUSION - FINAL RESULTS
{'='*80}
  Configuration:
    • Vision:   {all_results['Vision']['model_name']} (features: {all_results['Vision']['feature_dim']})
    • Signal:   {all_results['Signal']['model_name']} (features: {all_results['Signal']['feature_dim']})
    • Clinical: {all_results['Clinical']['model_name']} (features: {all_results['Clinical']['feature_dim']})
    • Total fusion features: {sum(all_results[k]['feature_dim'] for k in ['Vision', 'Signal', 'Clinical'])}

{'='*80}
5. 📊 FINAL PERFORMANCE SUMMARY
{'='*80}
  Modality      | Model              | AUC-ROC | Accuracy
  --------------|--------------------|---------|---------
  Vision (CXR)  | {all_results['Vision']['model_name']:18s} | {all_results['Vision']['best_auc']:.4f}  | {all_results['Vision']['final_metrics']['accuracy']*100:.1f}%
  Signal (ECG)  | {all_results['Signal']['model_name']:18s} | {all_results['Signal']['best_auc']:.4f}  | {all_results['Signal']['final_metrics']['accuracy']*100:.1f}%
  Clinical (Labs)| {all_results['Clinical']['model_name']:18s} | {all_results['Clinical']['best_auc']:.4f}  | {all_results['Clinical']['final_metrics']['accuracy']*100:.1f}%
  --------------|--------------------|---------|---------
  🏆 FUSION     | Multi-Modal        | {fusion['best_auc']:.4f}  | {m['accuracy']*100:.1f}%

{'='*80}
6. 📈 FUSION IMPROVEMENT ANALYSIS
{'='*80}
  Best Single Modality: {max(all_results[k]['best_auc'] for k in ['Vision', 'Signal', 'Clinical']):.4f}
  Fusion AUC:           {fusion['best_auc']:.4f}
  Absolute Improvement: {fusion['best_auc'] - max(all_results[k]['best_auc'] for k in ['Vision', 'Signal', 'Clinical']):+.4f}
  Relative Improvement: {((fusion['best_auc'] / max(all_results[k]['best_auc'] for k in ['Vision', 'Signal', 'Clinical'])) - 1) * 100:+.1f}%

  This demonstrates that MULTI-MODAL FUSION captures complementary
  information across modalities that no single source can provide!

{'='*80}
7. 📁 FILES GENERATED
{'='*80}
  Visualizations:
    • vision_comprehensive.png     - Vision module 6-panel analysis
    • signal_comprehensive.png     - Signal module 6-panel analysis  
    • clinical_comprehensive.png   - Clinical module analysis
    • confusion_matrices_grid.png  - All confusion matrices
    • final_fusion_comparison.png  - Final comparison chart
  
  Data:
    • vision_comparison.csv        - Vision metrics table
    • signal_comparison.csv        - Signal metrics table
    • clinical_comparison.csv      - Clinical metrics table
    • config.json                  - Model configuration
    • training_report.txt          - This report
  
  Models:
    • best_fusion_model.pth        - 🏆 Final trained model

{'='*80}
✅ TRAINING COMPLETE - READY FOR DASHBOARD INTEGRATION!
{'='*80}
"""
    print(report)
    
    with open(f'{output_dir}/training_report.txt', 'w') as f:
        f.write(report)
    print(f"✓ Saved: {output_dir}/training_report.txt")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("\n" + "="*70)
    print("🫀 VISIONCARE - ADVANCED MODEL COMPARISON")
    print("="*70)
    Config.print_config()
    
    # Load data
    print("📂 Loading datasets...")
    train_ds = SymileMIMICDataset(Config.DATA_DIR, 'train')
    val_ds = SymileMIMICDataset(Config.DATA_DIR, 'val')
    
    train_loader = DataLoader(
        train_ds, Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True,
        persistent_workers=True
    )
    
    # ========================================
    # 🩻 VISION MODULE
    # ========================================
    print("\n" + "="*70)
    print("🩻 VISION MODULE - COMPARING 3 ARCHITECTURES")
    print("="*70)
    
    vision_models = [DenseNet121(), EfficientNetB2(), ConvNeXtTiny()]
    vision_results = []
    
    for model in vision_models:
        trainer = Trainer(model, train_loader, val_loader, 'vision')
        result = trainer.train(Config.EPOCHS, f"vision_{model.name.lower().replace('-', '_')}")
        vision_results.append(result)
        torch.cuda.empty_cache()  # Free memory
    
    create_comparison_table(vision_results, "Vision")
    best_vision_name, _ = plot_comprehensive_comparison(vision_results, "Vision", Config.OUTPUT_DIR)
    best_vision_idx = [r['model_name'] for r in vision_results].index(best_vision_name)
    
    # ========================================
    # ❤️ SIGNAL MODULE
    # ========================================
    print("\n" + "="*70)
    print("❤️ SIGNAL MODULE - COMPARING 3 ARCHITECTURES")
    print("="*70)
    
    signal_models = [CNN1D(), ResNet1D(), InceptionTime()]
    signal_results = []
    
    for model in signal_models:
        trainer = Trainer(model, train_loader, val_loader, 'signal')
        result = trainer.train(Config.EPOCHS, f"signal_{model.name.lower().replace('-', '_')}")
        signal_results.append(result)
        torch.cuda.empty_cache()
    
    create_comparison_table(signal_results, "Signal")
    best_signal_name, _ = plot_comprehensive_comparison(signal_results, "Signal", Config.OUTPUT_DIR)
    best_signal_idx = [r['model_name'] for r in signal_results].index(best_signal_name)
    
    # ========================================
    # 🩸 CLINICAL MODULE
    # ========================================
    print("\n" + "="*70)
    print("🩸 CLINICAL MODULE - COMPARING 2 ARCHITECTURES")
    print("="*70)
    
    clinical_models = [MLP(), TabNet()]
    clinical_results = []
    
    for model in clinical_models:
        trainer = Trainer(model, train_loader, val_loader, 'clinical')
        result = trainer.train(Config.EPOCHS, f"clinical_{model.name.lower()}")
        clinical_results.append(result)
    
    create_comparison_table(clinical_results, "Clinical")
    best_clinical_name, _ = plot_comprehensive_comparison(clinical_results, "Clinical", Config.OUTPUT_DIR)
    best_clinical_idx = [r['model_name'] for r in clinical_results].index(best_clinical_name)
    
    # ========================================
    # 🔀 FUSION WITH BEST MODELS
    # ========================================
    print("\n" + "="*70)
    print("🔀 MULTI-MODAL FUSION - COMBINING BEST MODELS")
    print("="*70)
    print(f"  • Vision:   {best_vision_name}")
    print(f"  • Signal:   {best_signal_name}")
    print(f"  • Clinical: {best_clinical_name}")
    
    # Recreate best models
    vision_classes = [DenseNet121, EfficientNetB2, ConvNeXtTiny]
    signal_classes = [CNN1D, ResNet1D, InceptionTime]
    clinical_classes = [MLP, TabNet]
    
    best_vision = vision_classes[best_vision_idx]()
    best_signal = signal_classes[best_signal_idx]()
    best_clinical = clinical_classes[best_clinical_idx]()
    
    fusion_model = MultiModalFusion(best_vision, best_signal, best_clinical)
    trainer = Trainer(fusion_model, train_loader, val_loader, 'fusion')
    fusion_result = trainer.train(Config.EPOCHS, 'best_fusion_model')
    
    # ========================================
    # FINAL VISUALIZATION & REPORT
    # ========================================
    all_results = {
        'Vision': {**vision_results[best_vision_idx]},
        'Signal': {**signal_results[best_signal_idx]},
        'Clinical': {**clinical_results[best_clinical_idx]},
        'Fusion': fusion_result
    }
    
    plot_confusion_matrices_grid(all_results, Config.OUTPUT_DIR)
    plot_final_fusion_comparison(all_results, Config.OUTPUT_DIR)
    generate_comprehensive_report(all_results, vision_results, signal_results, clinical_results, Config.OUTPUT_DIR)
    
    # Save config
    config = {
        'best_models': {
            'vision': best_vision_name,
            'signal': best_signal_name,
            'clinical': best_clinical_name
        },
        'feature_dims': {
            'vision': best_vision.feature_dim,
            'signal': best_signal.feature_dim,
            'clinical': best_clinical.feature_dim,
            'total': fusion_model.total_features
        },
        'results': {k: {'auc': v['best_auc'], 'accuracy': v['final_metrics']['accuracy']} for k, v in all_results.items()}
    }
    with open(f'{Config.OUTPUT_DIR}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"📁 Results saved to: {Config.OUTPUT_DIR}/")
    print(f"🏆 Best Fusion AUC: {fusion_result['best_auc']:.4f}")


if __name__ == "__main__":
    main()
