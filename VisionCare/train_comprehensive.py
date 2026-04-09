"""
VisionCare - Multi-Modal Training with Model Comparison
Cardiovascular Disease Detection using CXR + ECG + Labs

Author: VisionCare Team - BTP Semester 7
Date: February 2026

This script provides comprehensive training with:
- Multiple models per modality (comparison)
- Best model selection per modality
- Multi-modal fusion training
- Performance visualization & reporting
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
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


print("✓ All libraries imported successfully!")

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    DATA_DIR = "/content/drive/MyDrive/symile-mimic"
    OUTPUT_DIR = "VisionCare_Results"
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATASET
# ============================================================
class SymileMIMICDataset(Dataset):
    """Multi-modal dataset with CXR, ECG, and Labs."""
    
    def __init__(self, data_dir, split='train'):
        self.split = split
        self.df = pd.read_csv(f"{data_dir}/{split}.csv")
        
        npy_dir = f"{data_dir}/data_npy/{split}"
        self.cxr = np.load(f"{npy_dir}/cxr_{split}.npy", mmap_mode='r')
        self.ecg = np.load(f"{npy_dir}/ecg_{split}.npy", mmap_mode='r')
        self.labs_pct = np.load(f"{npy_dir}/labs_percentiles_{split}.npy", mmap_mode='r')
        self.labs_miss = np.load(f"{npy_dir}/labs_missingness_{split}.npy", mmap_mode='r')
        
        if 'Cardiomegaly' in self.df.columns:
            labels = self.df['Cardiomegaly'].fillna(0).values
            self.labels = ((labels == 1.0) | (labels == -1.0)).astype(int)
        else:
            self.labels = np.zeros(len(self.df), dtype=int)
        
        print(f"✓ {split.upper()}: {len(self)} samples, {self.labels.mean()*100:.1f}% positive")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cxr = torch.tensor(self.cxr[idx].copy(), dtype=torch.float32)
        ecg = torch.tensor(self.ecg[idx].copy(), dtype=torch.float32).squeeze(0).transpose(0, 1)
        labs = torch.tensor(np.concatenate([self.labs_pct[idx], self.labs_miss[idx]]), dtype=torch.float32)
        return cxr, ecg, labs, int(self.labels[idx])


# ============================================================
# VISION MODELS (3 options for comparison)
# ============================================================
class DenseNet121CXR(nn.Module):
    """DenseNet-121 - Used in CheXNet (Stanford)"""
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "DenseNet-121", 1024
        self.backbone = models.densenet121(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 2))
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat), feat


class ResNet50CXR(nn.Module):
    """ResNet-50 - Standard baseline"""
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "ResNet-50", 2048
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 2))
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat), feat


class EfficientNetB0CXR(nn.Module):
    """EfficientNet-B0 - Efficient and modern"""
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "EfficientNet-B0", 1280
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 2))
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat), feat


# ============================================================
# SIGNAL MODELS (2 options for comparison)
# ============================================================
class ECG1DCNN(nn.Module):
    """1D-CNN - Fast and effective"""
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "1D-CNN", 256
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 15, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 11, padding=5), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 7, padding=3), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 2))
    
    def forward(self, x):
        feat = self.conv(x).squeeze(-1)
        return self.classifier(feat), feat


class ECGBiLSTM(nn.Module):
    """Bidirectional LSTM - Captures temporal dependencies"""
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "BiLSTM", 256
        self.downsample = nn.Sequential(nn.Conv1d(12, 64, 15, stride=5, padding=7), nn.BatchNorm1d(64), nn.ReLU())
        self.lstm = nn.LSTM(64, 128, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(256, 256)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 2))
    
    def forward(self, x):
        x = self.downsample(x).transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        feat = self.fc(torch.cat([h_n[-2], h_n[-1]], dim=1))
        return self.classifier(feat), feat


# ============================================================
# CLINICAL MODEL
# ============================================================
class ClinicalMLP(nn.Module):
    """MLP for blood lab values"""
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "MLP", 64
        self.encoder = nn.Sequential(nn.Linear(100, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 2))
    
    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat), feat


# ============================================================
# FUSION MODEL
# ============================================================
class VisionCareFusion(nn.Module):
    """Multi-modal fusion with best models from each modality"""
    def __init__(self, vision, signal, clinical):
        super().__init__()
        self.vision, self.signal, self.clinical = vision, signal, clinical
        total = vision.feature_dim + signal.feature_dim + clinical.feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(total, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
        )
        self.total_features = total
    
    def forward(self, cxr, ecg, labs):
        _, v = self.vision(cxr)
        _, s = self.signal(ecg)
        _, c = self.clinical(labs)
        return self.fusion(torch.cat([v, s, c], dim=1)), (v, s, c)


# ============================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================
def train_epoch(model, loader, optimizer, criterion, modality='fusion'):
    model.train()
    total_loss = 0
    for cxr, ecg, labs, y in tqdm(loader, desc="Training", leave=False):
        cxr, ecg, labs, y = cxr.to(Config.DEVICE), ecg.to(Config.DEVICE), labs.to(Config.DEVICE), y.to(Config.DEVICE)
        optimizer.zero_grad()
        if modality == 'fusion': logits, _ = model(cxr, ecg, labs)
        elif modality == 'vision': logits, _ = model(cxr)
        elif modality == 'signal': logits, _ = model(ecg)
        else: logits, _ = model(labs)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, modality='fusion'):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for cxr, ecg, labs, y in tqdm(loader, desc="Evaluating", leave=False):
            cxr, ecg, labs = cxr.to(Config.DEVICE), ecg.to(Config.DEVICE), labs.to(Config.DEVICE)
            if modality == 'fusion': logits, _ = model(cxr, ecg, labs)
            elif modality == 'vision': logits, _ = model(cxr)
            elif modality == 'signal': logits, _ = model(ecg)
            else: logits, _ = model(labs)
            all_probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(y.numpy())
    
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    return {
        'accuracy': accuracy_score(all_labels, preds),
        'precision': precision_score(all_labels, preds, zero_division=0),
        'recall': recall_score(all_labels, preds, zero_division=0),
        'f1': f1_score(all_labels, preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5,
        'probs': all_probs, 'labels': all_labels, 'preds': preds
    }


def train_model(model, train_loader, val_loader, epochs, lr, modality, save_name):
    model = model.to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    criterion = nn.CrossEntropyLoss()
    
    best_auc, best_epoch = 0, 0
    history = {'train_loss': [], 'val_auc': [], 'val_acc': []}
    
    print(f"\n{'='*60}")
    print(f"Training: {model.name}")
    print(f"{'='*60}")
    
    start = time.time()
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, modality)
        metrics = evaluate(model, val_loader, modality)
        scheduler.step(metrics['auc'])
        
        history['train_loss'].append(loss)
        history['val_auc'].append(metrics['auc'])
        history['val_acc'].append(metrics['accuracy'])
        
        marker = ""
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{Config.OUTPUT_DIR}/{save_name}.pth")
            marker = " ✅ Best!"
        
        print(f"Epoch {epoch+1:2d} | Loss: {loss:.4f} | AUC: {metrics['auc']:.4f} | Acc: {metrics['accuracy']*100:.1f}%{marker}")
    
    train_time = time.time() - start
    print(f"⏱️ Time: {train_time:.1f}s | Best AUC: {best_auc:.4f} (Epoch {best_epoch})")
    
    return {
        'model_name': model.name,
        'history': history,
        'best_auc': best_auc,
        'best_epoch': best_epoch,
        'train_time': train_time,
        'final_metrics': metrics
    }


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def plot_modality_comparison(results, modality_name, output_dir):
    """Plot comparison of models within a modality."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = [r['model_name'] for r in results]
    aucs = [r['best_auc'] for r in results]
    times = [r['train_time'] for r in results]
    colors = sns.color_palette("husl", len(results))
    
    # AUC Bar Chart
    bars = axes[0].bar(names, aucs, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('AUC-ROC', fontsize=12)
    axes[0].set_title(f'🏆 {modality_name} Model Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0.5, 1.0)
    for bar, auc in zip(bars, aucs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auc:.3f}', ha='center', fontweight='bold')
    
    # Mark best
    best_idx = np.argmax(aucs)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    
    # Training curves
    for r, color in zip(results, colors):
        axes[1].plot(r['history']['val_auc'], label=f"{r['model_name']}", linewidth=2, color=color)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation AUC')
    axes[1].set_title(f'📈 Training Progress', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{modality_name.lower()}_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/{modality_name.lower()}_comparison.png")
    plt.show()
    
    return names[best_idx], aucs[best_idx]


def plot_final_comparison(all_results, output_dir):
    """Plot final comparison of all modalities + fusion."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    names = list(all_results.keys())
    aucs = [all_results[n]['best_auc'] for n in names]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63'][:len(names)]
    
    # Bar chart
    bars = axes[0].bar(names, aucs, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    axes[0].set_title('🏆 Final Model Comparison (Best per Modality)', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0.5, 1.0)
    for bar, auc in zip(bars, aucs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auc:.3f}', ha='center', fontweight='bold', fontsize=11)
    
    # ROC Curves
    for (name, data), color in zip(all_results.items(), colors):
        m = data['final_metrics']
        fpr, tpr, _ = roc_curve(m['labels'], m['probs'])
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={data['best_auc']:.3f})", color=color, linewidth=2)
    
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('📊 ROC Curves', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/final_comparison.png")
    plt.show()


def generate_report(all_results, vision_comparison, signal_comparison, output_dir):
    """Generate comprehensive training report."""
    
    # Find best overall
    best_name = max(all_results.items(), key=lambda x: x[1]['best_auc'])[0]
    best_data = all_results[best_name]
    m = best_data['final_metrics']
    
    report = f"""
{'='*70}
VISIONCARE - MULTI-MODAL TRAINING REPORT
BTP Semester 7 - Cardiovascular Disease Detection
{'='*70}

📅 Training Date: February 2026

{'='*70}
1. VISION MODULE - MODEL COMPARISON
{'='*70}
"""
    for r in vision_comparison:
        star = "⭐" if r['model_name'] == all_results['Vision']['model_name'] else "  "
        report += f"  {star} {r['model_name']:20s}: AUC={r['best_auc']:.4f}, Time={r['train_time']:.1f}s\n"
    
    report += f"""
🏆 Best Vision Model: {all_results['Vision']['model_name']}

{'='*70}
2. SIGNAL MODULE - MODEL COMPARISON
{'='*70}
"""
    for r in signal_comparison:
        star = "⭐" if r['model_name'] == all_results['Signal']['model_name'] else "  "
        report += f"  {star} {r['model_name']:20s}: AUC={r['best_auc']:.4f}, Time={r['train_time']:.1f}s\n"
    
    report += f"""
🏆 Best Signal Model: {all_results['Signal']['model_name']}

{'='*70}
3. CLINICAL MODULE
{'='*70}
  ⭐ {all_results['Clinical']['model_name']:20s}: AUC={all_results['Clinical']['best_auc']:.4f}

{'='*70}
4. FUSION MODEL (ALL MODALITIES)
{'='*70}
  ⭐ Multi-Modal Fusion       : AUC={all_results['Fusion']['best_auc']:.4f}

{'='*70}
5. FINAL RESULTS SUMMARY
{'='*70}
  Vision (CXR):    {all_results['Vision']['best_auc']:.4f}
  Signal (ECG):    {all_results['Signal']['best_auc']:.4f}
  Clinical (Labs): {all_results['Clinical']['best_auc']:.4f}
  FUSION (All):    {all_results['Fusion']['best_auc']:.4f} 🏆

📈 Fusion Improvement over Best Single Modality:
  Best Single: {max(all_results['Vision']['best_auc'], all_results['Signal']['best_auc'], all_results['Clinical']['best_auc']):.4f}
  Fusion:      {all_results['Fusion']['best_auc']:.4f}
  Improvement: {((all_results['Fusion']['best_auc'] / max(all_results['Vision']['best_auc'], all_results['Signal']['best_auc'], all_results['Clinical']['best_auc'])) - 1) * 100:+.1f}%

{'='*70}
6. BEST MODEL DETAILED METRICS
{'='*70}
  Model:     {best_name} ({best_data['model_name']})
  Accuracy:  {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)
  Precision: {m['precision']:.4f}
  Recall:    {m['recall']:.4f}
  F1-Score:  {m['f1']:.4f}
  ROC-AUC:   {best_data['best_auc']:.4f}

{'='*70}
7. FILES GENERATED
{'='*70}
  • vision_comparison.png   - Vision model comparison
  • signal_comparison.png   - Signal model comparison  
  • final_comparison.png    - All modalities comparison
  • *.pth                   - Trained model weights

{'='*70}
✅ TRAINING COMPLETE - READY FOR DASHBOARD!
{'='*70}
"""
    print(report)
    
    with open(f'{output_dir}/training_report.txt', 'w') as f:
        f.write(report)
    print(f"✓ Saved: {output_dir}/training_report.txt")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("="*70)
    print("🫀 VISIONCARE - MULTI-MODAL CVD DETECTION")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n📂 Loading datasets...")
    train_ds = SymileMIMICDataset(Config.DATA_DIR, 'train')
    val_ds = SymileMIMICDataset(Config.DATA_DIR, 'val')
    train_loader = DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # ========================================
    # VISION MODULE - Compare 3 models
    # ========================================
    print("\n" + "="*70)
    print("🩻 VISION MODULE - COMPARING MODELS")
    print("="*70)
    
    vision_models = [DenseNet121CXR(), ResNet50CXR(), EfficientNetB0CXR()]
    vision_results = []
    
    for model in vision_models:
        result = train_model(model, train_loader, val_loader, Config.EPOCHS, Config.LEARNING_RATE, 'vision', f'vision_{model.name.lower().replace("-", "_")}')
        vision_results.append(result)
    
    best_vision_name, best_vision_auc = plot_modality_comparison(vision_results, "Vision", Config.OUTPUT_DIR)
    best_vision_idx = [r['model_name'] for r in vision_results].index(best_vision_name)
    
    # ========================================
    # SIGNAL MODULE - Compare 2 models
    # ========================================
    print("\n" + "="*70)
    print("❤️ SIGNAL MODULE - COMPARING MODELS")
    print("="*70)
    
    signal_models = [ECG1DCNN(), ECGBiLSTM()]
    signal_results = []
    
    for model in signal_models:
        result = train_model(model, train_loader, val_loader, Config.EPOCHS, Config.LEARNING_RATE, 'signal', f'signal_{model.name.lower().replace("-", "_")}')
        signal_results.append(result)
    
    best_signal_name, best_signal_auc = plot_modality_comparison(signal_results, "Signal", Config.OUTPUT_DIR)
    best_signal_idx = [r['model_name'] for r in signal_results].index(best_signal_name)
    
    # ========================================
    # CLINICAL MODULE
    # ========================================
    print("\n" + "="*70)
    print("🩸 CLINICAL MODULE")
    print("="*70)
    
    clinical_model = ClinicalMLP()
    clinical_result = train_model(clinical_model, train_loader, val_loader, Config.EPOCHS, Config.LEARNING_RATE, 'clinical', 'clinical_mlp')
    
    # ========================================
    # FUSION - Use best models from each
    # ========================================
    print("\n" + "="*70)
    print("🔀 FUSION MODULE - COMBINING BEST MODELS")
    print("="*70)
    print(f"  • Vision: {best_vision_name} (AUC: {best_vision_auc:.4f})")
    print(f"  • Signal: {best_signal_name} (AUC: {best_signal_auc:.4f})")
    print(f"  • Clinical: MLP (AUC: {clinical_result['best_auc']:.4f})")
    
    # Recreate best models for fusion
    vision_classes = [DenseNet121CXR, ResNet50CXR, EfficientNetB0CXR]
    signal_classes = [ECG1DCNN, ECGBiLSTM]
    
    best_vision = vision_classes[best_vision_idx]()
    best_signal = signal_classes[best_signal_idx]()
    best_clinical = ClinicalMLP()
    
    fusion_model = VisionCareFusion(best_vision, best_signal, best_clinical)
    fusion_result = train_model(fusion_model, train_loader, val_loader, Config.EPOCHS, Config.LEARNING_RATE, 'fusion', 'best_fusion_model')
    
    # ========================================
    # FINAL COMPARISON & REPORT
    # ========================================
    all_results = {
        'Vision': {**vision_results[best_vision_idx], 'model_name': best_vision_name},
        'Signal': {**signal_results[best_signal_idx], 'model_name': best_signal_name},
        'Clinical': clinical_result,
        'Fusion': fusion_result
    }
    
    plot_final_comparison(all_results, Config.OUTPUT_DIR)
    generate_report(all_results, vision_results, signal_results, Config.OUTPUT_DIR)
    
    # Save config
    config = {
        'best_models': {
            'vision': best_vision_name,
            'signal': best_signal_name,
            'clinical': 'MLP',
            'fusion_total_features': fusion_model.total_features
        },
        'results': {k: v['best_auc'] for k, v in all_results.items()}
    }
    with open(f'{Config.OUTPUT_DIR}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n🎉 TRAINING COMPLETE!")
    print(f"📁 All results saved to: {Config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
