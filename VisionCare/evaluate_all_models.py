# =====================================================================
# 📊 EVALUATE ALL SAVED MODELS - Get Full Metrics from Checkpoints
# =====================================================================
# This script loads ALL saved checkpoints from Google Drive and computes:
# - AUC-ROC, Precision, Recall, F1-Score, Specificity, Accuracy
# - No training needed! Just evaluation.
# =====================================================================

import os
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

import torchvision.models as models
from tqdm.auto import tqdm

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score, roc_curve
)

warnings.filterwarnings('ignore')

# ===================== ENVIRONMENT =====================

try:
    from google.colab import drive
    IN_COLAB = True
    drive.mount('/content/drive')
    print("✅ Running in Google Colab - Drive mounted!")
except:
    IN_COLAB = False
    print("💻 Running locally")


# ===================== CONFIG =====================

class Config:
    DATA_DIR = "/content/drive/MyDrive/symile-mimic" if IN_COLAB else "./data/symile-mimic"
    OUTPUT_DIR = "/content/drive/MyDrive/symile-mimic/MultiLabel_Results" if IN_COLAB else "MultiLabel_Results"
    CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
    
    LABELS = ['Cardiomegaly', 'Edema', 'Atelectasis', 'Pleural Effusion', 'Lung Opacity', 'No Finding']
    NUM_LABELS = 6
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== DATASET =====================

class SymileMIMICDataset(Dataset):
    def __init__(self, data_dir, split='val'):
        self.split = split
        csv_path = f"{data_dir}/{split}.csv"
        npy_dir = f"{data_dir}/data_npy/{split}"
        
        print(f"  📂 Loading {split} data...")
        self.df = pd.read_csv(csv_path)
        self.cxr = np.load(f"{npy_dir}/cxr_{split}.npy", mmap_mode='r')
        self.ecg = np.load(f"{npy_dir}/ecg_{split}.npy", mmap_mode='r')
        self.labs_percentiles = np.load(f"{npy_dir}/labs_percentiles_{split}.npy", mmap_mode='r')
        self.labs_missingness = np.load(f"{npy_dir}/labs_missingness_{split}.npy", mmap_mode='r')
        
        self.labels = np.zeros((len(self.df), Config.NUM_LABELS), dtype=np.float32)
        for i, name in enumerate(Config.LABELS):
            if name in self.df.columns:
                values = self.df[name].fillna(0).values
                self.labels[:, i] = ((values == 1.0) | (values == -1.0)).astype(float)
        print(f"  ✅ Loaded {len(self):,} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cxr = torch.from_numpy(self.cxr[idx].copy()).float()
        ecg = torch.from_numpy(self.ecg[idx].copy()).float()
        if ecg.dim() == 3: ecg = ecg.squeeze(0)
        if ecg.shape[0] != 12: ecg = ecg.transpose(0, 1)
        labs = torch.from_numpy(np.concatenate([self.labs_percentiles[idx], self.labs_missingness[idx]])).float()
        labels = torch.from_numpy(self.labels[idx]).float()
        return cxr, ecg, labs, labels


# ===================== ALL MODEL DEFINITIONS =====================

# --- Vision Models ---
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
        return self.classifier(features), features

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
        return self.classifier(features), features

class ConvNeXtTinyMultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "ConvNeXt-Tiny"
        self.feature_dim = 768
        self.backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Flatten(1), nn.LayerNorm(768), nn.Dropout(0.3), nn.Linear(768, num_labels))
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2: features = features.mean(dim=[-2, -1])
        return self.classifier(features), features

# --- Signal Models ---
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
        return self.classifier(features), features

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch)) if stride != 1 or in_ch != out_ch else None
    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + identity)

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
        x = self.layer3(self.layer2(self.layer1(self.stem(x))))
        features = self.avgpool(x).squeeze(-1)
        return self.classifier(features), features

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
        return F.relu(self.bn(self.conv(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1))))

class InceptionTimeMultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "InceptionTime"
        self.feature_dim = 256
        self.stem = nn.Sequential(nn.Conv1d(12, 32, 1), nn.BatchNorm1d(32), nn.ReLU())
        self.blocks = nn.Sequential(
            InceptionBlock(32, 32), nn.MaxPool1d(2), InceptionBlock(32, 64), nn.MaxPool1d(2),
            InceptionBlock(64, 128), nn.MaxPool1d(2), InceptionBlock(128, 256), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_labels))
    def forward(self, x):
        features = self.blocks(self.stem(x)).squeeze(-1)
        return self.classifier(features), features

# --- Clinical Models ---
class MLPMultiLabel(nn.Module):
    def __init__(self, input_dim=100, num_labels=6):
        super().__init__()
        self.name = "MLP"
        self.feature_dim = 64
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_labels))
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features), features

class TabNetMultiLabel(nn.Module):
    def __init__(self, input_dim=100, n_steps=3, hidden_dim=64, num_labels=6):
        super().__init__()
        self.name = "TabNet"
        self.feature_dim = 64
        self.bn = nn.BatchNorm1d(input_dim)
        self.initial_fc = nn.Linear(input_dim, hidden_dim)
        self.attention_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.BatchNorm1d(input_dim), nn.Softmax(dim=-1)) for _ in range(n_steps)])
        self.fc_layers = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()) for _ in range(n_steps)])
        self.final_fc = nn.Linear(hidden_dim, self.feature_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.feature_dim, num_labels))
    def forward(self, x):
        x = self.bn(x)
        h = F.relu(self.initial_fc(x))
        agg = torch.zeros_like(h)
        for attn, fc in zip(self.attention_layers, self.fc_layers):
            step_out = fc(x * attn(h))
            agg, h = agg + step_out, step_out
        return self.classifier(self.final_fc(agg)), self.final_fc(agg)


# ===================== COMPREHENSIVE METRICS =====================

def compute_all_metrics(labels, probs, threshold=0.5):
    """Compute all metrics including Accuracy, Precision, Recall, F1, Specificity."""
    preds = (probs > threshold).astype(int)
    results = {'per_class': {}, 'macro': {}}
    
    metrics_lists = defaultdict(list)
    
    for i, name in enumerate(Config.LABELS):
        y_true, y_pred, y_prob = labels[:, i], preds[:, i], probs[:, i]
        
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) >= 2 else 0.5
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        results['per_class'][name] = {
            'auc': auc, 'precision': precision, 'recall': recall,
            'f1': f1, 'specificity': specificity, 'accuracy': accuracy
        }
        
        for k, v in results['per_class'][name].items():
            metrics_lists[k].append(v)
    
    results['macro'] = {k: np.mean(v) for k, v in metrics_lists.items()}
    return results


# ===================== EVALUATE MODEL =====================

@torch.no_grad()
def evaluate_model(model, dataloader, modality):
    """Run inference and compute metrics."""
    model.eval()
    all_probs, all_labels = [], []
    
    for cxr, ecg, labs, labels in tqdm(dataloader, desc=f"Evaluating {model.name}", leave=False):
        if modality == 'vision':
            x = cxr.to(Config.DEVICE)
        elif modality == 'signal':
            x = ecg.to(Config.DEVICE)
        else:
            x = labs.to(Config.DEVICE)
        
        logits, _ = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    return compute_all_metrics(all_labels, all_probs)


# ===================== MAIN EVALUATION =====================

def main():
    print("\n" + "="*80)
    print("📊 EVALUATE ALL SAVED MODELS - Get Full Metrics")
    print("="*80)
    
    # Check checkpoint directory
    if not os.path.exists(Config.CHECKPOINT_DIR):
        print(f"❌ Checkpoint directory not found: {Config.CHECKPOINT_DIR}")
        return
    
    # List available checkpoints
    print(f"\n📁 Available checkpoints in: {Config.CHECKPOINT_DIR}")
    checkpoints = [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.endswith('.pth')]
    for f in checkpoints:
        size = os.path.getsize(f"{Config.CHECKPOINT_DIR}/{f}") / (1024*1024)
        print(f"   • {f} ({size:.1f} MB)")
    
    # Map checkpoint files to model classes
    MODEL_MAP = {
        'vision_densenet121': ('vision', DenseNet121MultiLabel),
        'vision_efficientnetb2': ('vision', EfficientNetB2MultiLabel),
        'vision_convnexttiny': ('vision', ConvNeXtTinyMultiLabel),
        'signal_1dcnn': ('signal', CNN1DMultiLabel),
        'signal_resnet1d': ('signal', ResNet1DMultiLabel),
        'signal_inceptiontime': ('signal', InceptionTimeMultiLabel),
        'clinical_mlp': ('clinical', MLPMultiLabel),
        'clinical_tabnet': ('clinical', TabNetMultiLabel),
    }
    
    # Load validation data
    print("\n📂 Loading validation data...")
    val_ds = SymileMIMICDataset(Config.DATA_DIR, 'val')
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Evaluate each checkpoint
    all_results = {}
    
    print("\n" + "="*80)
    print("🔬 EVALUATING MODELS...")
    print("="*80)
    
    for ckpt_file in checkpoints:
        name = ckpt_file.replace('.pth', '')
        
        if name not in MODEL_MAP:
            print(f"⚠️ Skipping unknown checkpoint: {ckpt_file}")
            continue
        
        modality, ModelClass = MODEL_MAP[name]
        
        print(f"\n📥 Loading: {name}...")
        
        try:
            # Create model and load weights
            model = ModelClass().to(Config.DEVICE)
            checkpoint = torch.load(f"{Config.CHECKPOINT_DIR}/{ckpt_file}", 
                                   weights_only=False, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            saved_auc = checkpoint.get('best_auc', 'N/A')
            epoch = checkpoint.get('epoch', 'N/A')
            
            print(f"   ✅ Loaded (Epoch {epoch}, Saved AUC: {saved_auc})")
            
            # Evaluate
            metrics = evaluate_model(model, val_loader, modality)
            all_results[name] = metrics
            
            # Print summary
            m = metrics['macro']
            print(f"   📊 Macro Metrics: AUC={m['auc']:.4f} | Prec={m['precision']:.4f} | "
                  f"Recall={m['recall']:.4f} | F1={m['f1']:.4f} | Acc={m['accuracy']:.4f}")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # ===================== PRINT FULL RESULTS TABLE =====================
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE RESULTS - ALL MODELS")
    print("="*100)
    
    # Header
    print(f"\n{'Model':<25} | {'AUC':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'Spec':>7} | {'Acc':>7}")
    print("-"*100)
    
    # Group by modality
    for modality in ['vision', 'signal', 'clinical']:
        mod_results = {k: v for k, v in all_results.items() if k.startswith(modality)}
        if mod_results:
            print(f"\n--- {modality.upper()} MODELS ---")
            for name, metrics in sorted(mod_results.items()):
                m = metrics['macro']
                print(f"{name:<25} | {m['auc']:>7.4f} | {m['precision']:>7.4f} | "
                      f"{m['recall']:>7.4f} | {m['f1']:>7.4f} | {m['specificity']:>7.4f} | {m['accuracy']:>7.4f}")
    
    # ===================== SAVE RESULTS =====================
    
    # Convert to JSON-serializable format
    results_json = {}
    for name, metrics in all_results.items():
        results_json[name] = {
            'macro': metrics['macro'],
            'per_class': metrics['per_class']
        }
    
    output_file = f"{Config.OUTPUT_DIR}/all_models_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")
    
    # ===================== CREATE COMPARISON CHART =====================
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data
    names = list(all_results.keys())
    aucs = [all_results[n]['macro']['auc'] for n in names]
    f1s = [all_results[n]['macro']['f1'] for n in names]
    
    colors = []
    for n in names:
        if 'vision' in n: colors.append('#3498db')
        elif 'signal' in n: colors.append('#e74c3c')
        else: colors.append('#9b59b6')
    
    # AUC Chart
    bars = axes[0].barh(names, aucs, color=colors, edgecolor='black')
    axes[0].set_xlabel('Macro AUC-ROC', fontsize=12)
    axes[0].set_title('📊 All Models: AUC Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0.4, 0.9)
    for bar, auc in zip(bars, aucs):
        axes[0].text(auc + 0.01, bar.get_y() + bar.get_height()/2, f'{auc:.4f}', va='center')
    
    # F1 Chart
    bars = axes[1].barh(names, f1s, color=colors, edgecolor='black')
    axes[1].set_xlabel('Macro F1-Score', fontsize=12)
    axes[1].set_title('📊 All Models: F1 Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 0.6)
    for bar, f1 in zip(bars, f1s):
        axes[1].text(f1 + 0.01, bar.get_y() + bar.get_height()/2, f'{f1:.4f}', va='center')
    
    plt.tight_layout()
    chart_file = f"{Config.OUTPUT_DIR}/all_models_comparison.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved to: {chart_file}")
    plt.show()
    
    print("\n✨ EVALUATION COMPLETE! ✨")


if __name__ == "__main__":
    main()
