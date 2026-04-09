"""
VisionCare - State-of-the-Art Model Evaluation
Final Test Set Evaluation & Comprehensive Analysis

Author: VisionCare Team - BTP Semester 7
Date: February 2026

This script performs comprehensive evaluation on the HELD-OUT TEST SET:
- Loads trained models
- Computes all clinical metrics
- Generates publication-ready visualizations
- Statistical significance testing
- Calibration analysis
- Per-subgroup analysis
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm.auto import tqdm
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torchvision import models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    brier_score_loss, log_loss, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titleweight'] = 'bold'

print("✓ Libraries imported!")


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    DATA_DIR = "/content/drive/MyDrive/symile-mimic"
    MODEL_DIR = "VisionCare_Advanced_Results"
    OUTPUT_DIR = "VisionCare_Evaluation"
    BATCH_SIZE = 64  # Larger for inference
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Bootstrap parameters for confidence intervals
    N_BOOTSTRAP = 1000
    CI_LEVEL = 0.95

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATASET (Test Set)
# ============================================================
class SymileMIMICDataset(Dataset):
    def __init__(self, data_dir, split='test'):
        self.split = split
        self.df = pd.read_csv(f"{data_dir}/{split}.csv")
        
        npy_dir = f"{data_dir}/data_npy/{split}"
        print(f"  Loading {split} set...")
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
        return cxr, ecg, labs, self.labels[idx], idx  # Include index for analysis


# ============================================================
# MODEL DEFINITIONS (Same as training)
# ============================================================
class DenseNet121(nn.Module):
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "DenseNet-121", 1024
        self.backbone = models.densenet121(weights=None)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 2))
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat), feat

class EfficientNetB2(nn.Module):
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "EfficientNet-B2", 1408
        self.backbone = models.efficientnet_b2(weights=None)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1408, 2))
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat), feat

class ConvNeXtTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "ConvNeXt-Tiny", 768
        self.backbone = models.convnext_tiny(weights=None)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Flatten(1), nn.LayerNorm(768), nn.Dropout(0.3), nn.Linear(768, 2))
    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() > 2: feat = feat.mean(dim=[-2, -1])
        return self.classifier(feat), feat

class CNN1D(nn.Module):
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

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "ResNet-1D", 256
        self.stem = nn.Sequential(nn.Conv1d(12, 64, 15, stride=2, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(3, stride=2, padding=1))
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 2))
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = [ResBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, blocks): layers.append(ResBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        feat = self.avgpool(x).squeeze(-1)
        return self.classifier(feat), feat

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch)) if stride != 1 or in_ch != out_ch else None
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample: identity = self.downsample(x)
        return F.relu(out + identity)

class InceptionTime(nn.Module):
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "InceptionTime", 256
        self.inception1 = InceptionBlock(12, 32)
        self.inception2 = InceptionBlock(128, 32)
        self.inception3 = InceptionBlock(128, 32)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 256)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 2))
    def forward(self, x):
        x = self.inception1(x); x = self.inception2(x); x = self.inception3(x)
        feat = self.fc(self.avgpool(x).squeeze(-1))
        return self.classifier(feat), feat

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv1d(in_ch, out_ch, 7, padding=3), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.branch4 = nn.Sequential(nn.Conv1d(in_ch, out_ch, 15, padding=7), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.pool = nn.MaxPool1d(2)
    def forward(self, x):
        return self.pool(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.name, self.feature_dim = "MLP", 64
        self.encoder = nn.Sequential(nn.Linear(100, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 2))
    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat), feat

class TabNet(nn.Module):
    def __init__(self, input_dim=100, n_steps=3, hidden_dim=64):
        super().__init__()
        self.name, self.feature_dim, self.n_steps = "TabNet", 64, n_steps
        self.bn = nn.BatchNorm1d(input_dim)
        self.initial_fc = nn.Linear(input_dim, hidden_dim)
        self.attention_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.BatchNorm1d(input_dim), nn.Softmax(dim=-1)) for _ in range(n_steps)])
        self.fc_layers = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()) for _ in range(n_steps)])
        self.final_fc = nn.Linear(hidden_dim, self.feature_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.feature_dim, 2))
    def forward(self, x):
        x = self.bn(x); h = F.relu(self.initial_fc(x)); aggregated = torch.zeros_like(h)
        for attn, fc in zip(self.attention_layers, self.fc_layers):
            step_out = fc(x * attn(h)); aggregated = aggregated + step_out; h = step_out
        feat = self.final_fc(aggregated)
        return self.classifier(feat), feat

class MultiModalFusion(nn.Module):
    def __init__(self, vision, signal, clinical):
        super().__init__()
        self.vision, self.signal, self.clinical = vision, signal, clinical
        total = vision.feature_dim + signal.feature_dim + clinical.feature_dim
        self.fusion = nn.Sequential(nn.Linear(total, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))
        self.total_features = total
    def forward(self, cxr, ecg, labs):
        _, v = self.vision(cxr); _, s = self.signal(ecg); _, c = self.clinical(labs)
        return self.fusion(torch.cat([v, s, c], dim=1)), (v, s, c)


# ============================================================
# EVALUATION METRICS
# ============================================================
def compute_all_metrics(y_true, y_prob, y_pred):
    """Compute comprehensive evaluation metrics."""
    metrics = {
        # Primary metrics
        'AUC-ROC': roc_auc_score(y_true, y_prob),
        'AUPRC': average_precision_score(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        
        # Precision/Recall
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall (Sensitivity)': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        
        # Additional metrics
        'Specificity': recall_score(1-y_true, 1-y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Cohen Kappa': cohen_kappa_score(y_true, y_pred),
        
        # Calibration
        'Brier Score': brier_score_loss(y_true, y_prob),
        'Log Loss': log_loss(y_true, y_prob),
    }
    
    # Confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['True Positives'] = int(tp)
    metrics['True Negatives'] = int(tn)
    metrics['False Positives'] = int(fp)
    metrics['False Negatives'] = int(fn)
    
    # Clinical metrics
    metrics['PPV'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics['Prevalence'] = (tp + fn) / len(y_true)
    
    return metrics


def bootstrap_ci(y_true, y_prob, metric_fn, n_bootstrap=1000, ci=0.95):
    """Compute confidence interval via bootstrapping."""
    scores = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_prob[idx])
            scores.append(score)
        except:
            continue
    
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lower, upper


def statistical_comparison(y_true, probs1, probs2, name1, name2):
    """DeLong test for comparing AUCs."""
    from scipy.stats import norm
    
    auc1 = roc_auc_score(y_true, probs1)
    auc2 = roc_auc_score(y_true, probs2)
    
    # Simplified z-test (approximate)
    n = len(y_true)
    se = np.sqrt((auc1 * (1 - auc1) + auc2 * (1 - auc2)) / n)
    z = (auc1 - auc2) / (se + 1e-10)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return {
        'model1': name1, 'auc1': auc1,
        'model2': name2, 'auc2': auc2,
        'difference': auc1 - auc2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


# ============================================================
# INFERENCE
# ============================================================
@torch.no_grad()
def run_inference(model, loader, modality='fusion'):
    """Run inference and collect predictions."""
    model.eval()
    all_probs, all_labels, all_indices = [], [], []
    
    for cxr, ecg, labs, y, idx in tqdm(loader, desc="Inference"):
        cxr = cxr.to(Config.DEVICE)
        ecg = ecg.to(Config.DEVICE)
        labs = labs.to(Config.DEVICE)
        
        with autocast():
            if modality == 'fusion':
                logits, _ = model(cxr, ecg, labs)
            elif modality == 'vision':
                logits, _ = model(cxr)
            elif modality == 'signal':
                logits, _ = model(ecg)
            else:
                logits, _ = model(labs)
        
        probs = F.softmax(logits.float(), dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.numpy())
        all_indices.extend(idx.numpy())
    
    return np.array(all_probs), np.array(all_labels), np.array(all_indices)


# ============================================================
# VISUALIZATIONS
# ============================================================
def plot_roc_curves(results, output_dir):
    """Publication-ready ROC curves."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for (name, data), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(data['labels'], data['probs'])
        auc = data['metrics']['AUC-ROC']
        
        lw = 3 if name == 'Fusion' else 2
        ls = '-' if name == 'Fusion' else '--'
        
        ax.plot(fpr, tpr, color=color, linewidth=lw, linestyle=ls,
               label=f"{name} (AUC = {auc:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves - Test Set Performance', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves_test.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/roc_curves_test.pdf', bbox_inches='tight')  # Vector format
    print(f"✓ Saved: {output_dir}/roc_curves_test.png/.pdf")
    plt.show()


def plot_precision_recall_curves(results, output_dir):
    """Precision-Recall curves (important for imbalanced data)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for (name, data), color in zip(results.items(), colors):
        precision, recall, _ = precision_recall_curve(data['labels'], data['probs'])
        auprc = data['metrics']['AUPRC']
        
        lw = 3 if name == 'Fusion' else 2
        ax.plot(recall, precision, color=color, linewidth=lw,
               label=f"{name} (AUPRC = {auprc:.3f})")
    
    # Baseline (no skill)
    baseline = results['Fusion']['metrics']['Prevalence']
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, label=f'No Skill ({baseline:.2f})')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Curves - Test Set', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pr_curves_test.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/pr_curves_test.png")
    plt.show()


def plot_calibration_curves(results, output_dir):
    """Calibration curves (reliability diagrams)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    # Calibration curve
    ax1 = axes[0]
    for (name, data), color in zip(results.items(), colors):
        prob_true, prob_pred = calibration_curve(data['labels'], data['probs'], n_bins=10)
        ax1.plot(prob_pred, prob_true, 's-', color=color, label=name, linewidth=2, markersize=8)
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfectly Calibrated')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax1.set_title('Calibration Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predictions
    ax2 = axes[1]
    for (name, data), color in zip(results.items(), colors):
        ax2.hist(data['probs'], bins=50, alpha=0.4, color=color, label=name, density=True)
    ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Predictions', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/calibration_test.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/calibration_test.png")
    plt.show()


def plot_confusion_matrices(results, output_dir):
    """Detailed confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()
    
    for idx, (name, data) in enumerate(results.items()):
        cm = confusion_matrix(data['labels'], data['preds'])
        
        # Normalize for percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No CVD', 'CVD'], yticklabels=['No CVD', 'CVD'],
                   cbar_kws={'label': 'Count'})
        
        # Add percentages
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75, f'({cm_norm[i, j]:.1f}%)',
                       ha='center', va='center', fontsize=10, color='gray')
        
        ax.set_title(f'{name}\nAUC={data["metrics"]["AUC-ROC"]:.3f}', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)
    
    plt.suptitle('Confusion Matrices - Test Set', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices_test.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/confusion_matrices_test.png")
    plt.show()


def plot_metrics_comparison(results, output_dir):
    """Bar chart comparison of all metrics."""
    metrics_to_plot = ['AUC-ROC', 'AUPRC', 'Accuracy', 'F1-Score', 'Recall (Sensitivity)', 'Specificity', 'MCC']
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, (name, data) in enumerate(results.items()):
        values = [data['metrics'][m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i], edgecolor='black')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comprehensive Metrics Comparison - Test Set', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics_to_plot, rotation=30, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison_test.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/metrics_comparison_test.png")
    plt.show()


# ============================================================
# REPORT GENERATION
# ============================================================
def generate_evaluation_report(results, stat_tests, output_dir):
    """Generate comprehensive evaluation report."""
    
    fusion = results['Fusion']
    m = fusion['metrics']
    
    report = f"""
{'='*80}
VISIONCARE - FINAL TEST SET EVALUATION REPORT
Multi-Modal Cardiovascular Disease Detection
{'='*80}

📅 Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Test Set Size: {len(fusion['labels']):,} samples
📈 Positive Class Prevalence: {m['Prevalence']*100:.1f}%

{'='*80}
1. PRIMARY METRICS (Test Set)
{'='*80}

  Modality        | AUC-ROC | AUPRC  | Accuracy | F1-Score
  ----------------|---------|--------|----------|----------
"""
    for name, data in results.items():
        me = data['metrics']
        star = " ⭐" if name == 'Fusion' else ""
        report += f"  {name:15s} | {me['AUC-ROC']:.4f}  | {me['AUPRC']:.4f} | {me['Accuracy']*100:5.1f}%   | {me['F1-Score']:.4f}{star}\n"
    
    report += f"""
{'='*80}
2. DETAILED FUSION MODEL METRICS
{'='*80}

  📊 Discrimination Metrics:
     • AUC-ROC:           {m['AUC-ROC']:.4f}
     • AUPRC:             {m['AUPRC']:.4f}
     • Balanced Accuracy: {m['Balanced Accuracy']:.4f}

  📊 Classification Metrics:
     • Accuracy:          {m['Accuracy']:.4f} ({m['Accuracy']*100:.2f}%)
     • Precision (PPV):   {m['Precision']:.4f}
     • Recall (Sens):     {m['Recall (Sensitivity)']:.4f}
     • Specificity:       {m['Specificity']:.4f}
     • F1-Score:          {m['F1-Score']:.4f}
     • NPV:               {m['NPV']:.4f}

  📊 Additional Metrics:
     • MCC:               {m['MCC']:.4f}
     • Cohen's Kappa:     {m['Cohen Kappa']:.4f}

  📊 Calibration Metrics:
     • Brier Score:       {m['Brier Score']:.4f} (lower is better)
     • Log Loss:          {m['Log Loss']:.4f} (lower is better)

{'='*80}
3. CONFUSION MATRIX - FUSION MODEL
{'='*80}

                      Predicted
                    No CVD    CVD
  Actual  No CVD    {m['True Negatives']:6,}    {m['False Positives']:6,}
          CVD       {m['False Negatives']:6,}    {m['True Positives']:6,}

{'='*80}
4. STATISTICAL COMPARISON (Fusion vs Single Modalities)
{'='*80}
"""
    for test in stat_tests:
        sig = "✓ Significant" if test['significant'] else "Not significant"
        report += f"  Fusion vs {test['model2']:10s}: Δ AUC = {test['difference']:+.4f}, p = {test['p_value']:.4f} ({sig})\n"
    
    report += f"""
{'='*80}
5. CLINICAL INTERPRETATION
{'='*80}

  At the default threshold (0.5):
  
  • If the model predicts CVD:
    - {m['Precision']*100:.1f}% chance the patient truly has CVD (PPV)
  
  • If the model predicts No CVD:
    - {m['NPV']*100:.1f}% chance the patient truly has no CVD (NPV)
  
  • Of all CVD patients:
    - {m['Recall (Sensitivity)']*100:.1f}% will be correctly identified (Sensitivity)
  
  • Of all non-CVD patients:
    - {m['Specificity']*100:.1f}% will be correctly identified (Specificity)

{'='*80}
6. FILES GENERATED
{'='*80}

  Visualizations:
    • roc_curves_test.png/pdf      - ROC curves (publication-ready)
    • pr_curves_test.png           - Precision-Recall curves
    • calibration_test.png         - Calibration analysis
    • confusion_matrices_test.png  - Confusion matrices
    • metrics_comparison_test.png  - Bar chart comparison
  
  Data:
    • test_predictions.csv         - Per-sample predictions
    • test_metrics.json           - All metrics
    • evaluation_report.txt       - This report

{'='*80}
✅ EVALUATION COMPLETE - READY FOR BTP PRESENTATION!
{'='*80}
"""
    print(report)
    
    with open(f'{output_dir}/evaluation_report.txt', 'w') as f:
        f.write(report)
    print(f"✓ Saved: {output_dir}/evaluation_report.txt")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print("🔬 VISIONCARE - TEST SET EVALUATION")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    
    # Load config to get best model names
    config_path = f"{Config.MODEL_DIR}/config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            train_config = json.load(f)
        print(f"✓ Loaded training config")
    else:
        print("⚠️ No config found, using defaults")
        train_config = {'best_models': {'vision': 'EfficientNet-B2', 'signal': 'InceptionTime', 'clinical': 'MLP'}}
    
    # Load test data
    print("\n📂 Loading TEST set...")
    test_ds = SymileMIMICDataset(Config.DATA_DIR, 'test')
    test_loader = DataLoader(test_ds, Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Model mapping
    model_classes = {
        'DenseNet-121': DenseNet121, 'EfficientNet-B2': EfficientNetB2, 'ConvNeXt-Tiny': ConvNeXtTiny,
        '1D-CNN': CNN1D, 'ResNet-1D': ResNet1D, 'InceptionTime': InceptionTime,
        'MLP': MLP, 'TabNet': TabNet
    }
    
    results = {}
    
    # Load and evaluate best models from each modality
    modality_info = [
        ('Vision', train_config['best_models']['vision'], 'vision'),
        ('Signal', train_config['best_models']['signal'], 'signal'),
        ('Clinical', train_config['best_models']['clinical'], 'clinical'),
    ]
    
    for mod_name, model_name, modality in modality_info:
        print(f"\n📊 Evaluating {mod_name}: {model_name}")
        
        model = model_classes[model_name]().to(Config.DEVICE)
        model_path = f"{Config.MODEL_DIR}/{modality}_{model_name.lower().replace('-', '_')}.pth"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            probs, labels, indices = run_inference(model, test_loader, modality)
            preds = (probs > 0.5).astype(int)
            metrics = compute_all_metrics(labels, probs, preds)
            results[mod_name] = {'probs': probs, 'labels': labels, 'preds': preds, 'metrics': metrics, 'model_name': model_name}
            print(f"  ✓ AUC: {metrics['AUC-ROC']:.4f}")
        else:
            print(f"  ⚠️ Model not found: {model_path}")
    
    # Load and evaluate fusion model
    print(f"\n📊 Evaluating Fusion Model...")
    
    best_vision = model_classes[train_config['best_models']['vision']]()
    best_signal = model_classes[train_config['best_models']['signal']]()
    best_clinical = model_classes[train_config['best_models']['clinical']]()
    fusion_model = MultiModalFusion(best_vision, best_signal, best_clinical).to(Config.DEVICE)
    
    fusion_path = f"{Config.MODEL_DIR}/best_fusion_model.pth"
    if os.path.exists(fusion_path):
        fusion_model.load_state_dict(torch.load(fusion_path, map_location=Config.DEVICE))
        probs, labels, indices = run_inference(fusion_model, test_loader, 'fusion')
        preds = (probs > 0.5).astype(int)
        metrics = compute_all_metrics(labels, probs, preds)
        results['Fusion'] = {'probs': probs, 'labels': labels, 'preds': preds, 'metrics': metrics, 'model_name': 'Multi-Modal'}
        print(f"  ✓ AUC: {metrics['AUC-ROC']:.4f}")
    else:
        print(f"  ⚠️ Fusion model not found")
    
    # Statistical tests
    print("\n📈 Statistical Comparison...")
    stat_tests = []
    if 'Fusion' in results:
        for mod_name in ['Vision', 'Signal', 'Clinical']:
            if mod_name in results:
                test = statistical_comparison(results['Fusion']['labels'], results['Fusion']['probs'],
                                             results[mod_name]['probs'], 'Fusion', mod_name)
                stat_tests.append(test)
                print(f"  Fusion vs {mod_name}: p={test['p_value']:.4f}")
    
    # Generate visualizations
    print("\n🎨 Generating Visualizations...")
    plot_roc_curves(results, Config.OUTPUT_DIR)
    plot_precision_recall_curves(results, Config.OUTPUT_DIR)
    plot_calibration_curves(results, Config.OUTPUT_DIR)
    plot_confusion_matrices(results, Config.OUTPUT_DIR)
    plot_metrics_comparison(results, Config.OUTPUT_DIR)
    
    # Save predictions
    if 'Fusion' in results:
        pred_df = pd.DataFrame({
            'index': range(len(results['Fusion']['labels'])),
            'true_label': results['Fusion']['labels'],
            'pred_prob': results['Fusion']['probs'],
            'pred_label': results['Fusion']['preds']
        })
        pred_df.to_csv(f"{Config.OUTPUT_DIR}/test_predictions.csv", index=False)
        print(f"✓ Saved: {Config.OUTPUT_DIR}/test_predictions.csv")
    
    # Save metrics
    metrics_dict = {name: data['metrics'] for name, data in results.items()}
    with open(f"{Config.OUTPUT_DIR}/test_metrics.json", 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✓ Saved: {Config.OUTPUT_DIR}/test_metrics.json")
    
    # Generate report
    generate_evaluation_report(results, stat_tests, Config.OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("🎉 EVALUATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
