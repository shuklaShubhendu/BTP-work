"""
VisionCare - FUSION MODEL Training
===================================
Loads the BEST models from Vision, Signal, Clinical and trains Fusion.

PREREQUISITES: Run the other 3 scripts first to generate:
  - checkpoints/vision_best.pth
  - checkpoints/signal_best.pth
  - checkpoints/clinical_best.pth

USAGE: Copy to Colab notebook, run all cells.
OUTPUT: Final fusion model and comprehensive report.
"""

# ===================== CELL 1: SETUP & MOUNT DRIVE =====================
import os

from google.colab import drive
drive.mount('/content/drive')

# ========== AUTO-DETECT DATA FOLDER ==========
POSSIBLE_PATHS = [
    "/content/drive/MyDrive/symile-mimic",
    "/content/drive/My Drive/symile-mimic", 
    "/content/drive/Shareddrives/symile-mimic",
    "/content/drive/MyDrive/symile_mimic",
]

SHARED_FOLDER = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path) and os.path.exists(f"{path}/train.csv"):
        SHARED_FOLDER = path
        break

if SHARED_FOLDER is None:
    mydrive = "/content/drive/MyDrive"
    if os.path.exists(mydrive):
        folders = [f for f in os.listdir(mydrive) if 'symile' in f.lower() or 'mimic' in f.lower()]
        for folder in folders:
            test_path = f"{mydrive}/{folder}"
            if os.path.exists(f"{test_path}/train.csv"):
                SHARED_FOLDER = test_path
                break
    
    if SHARED_FOLDER is None:
        print("❌ Data NOT found! Add shortcut: https://drive.google.com/drive/folders/12Ris_8Kav_MQoO5Luxw3Ylq33m-ewX_p")
        raise FileNotFoundError("Data folder not found")

OUTPUT_DIR = f"{SHARED_FOLDER}/MultiLabel_Results"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"✅ Data folder: {SHARED_FOLDER}")
print(f"💾 Checkpoints: {CHECKPOINT_DIR}")

# Check if all best models exist
required_files = ['vision_best.pth', 'signal_best.pth', 'clinical_best.pth']
missing = [f for f in required_files if not os.path.exists(f"{CHECKPOINT_DIR}/{f}")]

if missing:
    print(f"❌ Missing checkpoints: {missing}")
    print("   Run the Vision, Signal, and Clinical scripts first!")
else:
    print("✅ All pretrained models found!")

# ===================== CELL 2: IMPORTS =====================
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, roc_curve

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# ===================== CELL 3: CONFIG =====================
class Config:
    DATA_DIR = SHARED_FOLDER
    LABELS = ['Cardiomegaly', 'Edema', 'Atelectasis', 'Pleural Effusion', 'Lung Opacity', 'No Finding']
    NUM_LABELS = 6
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    EPOCHS = 20
    LEARNING_RATE = 5e-5  # Lower LR for fine-tuning
    WEIGHT_DECAY = 1e-5
    USE_AMP = True

# ===================== CELL 4: DATASET =====================
class MultiModalDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.split = split
        csv_path = f"{data_dir}/{split}.csv"
        npy_dir = f"{data_dir}/data_npy/{split}"
        
        print(f"  📂 Loading {split} data...")
        self.df = pd.read_csv(csv_path)
        
        # Load all modalities
        self.cxr = np.load(f"{npy_dir}/cxr_{split}.npy", mmap_mode='r')
        self.ecg = np.load(f"{npy_dir}/ecg_{split}.npy", mmap_mode='r')
        self.labs_pct = np.load(f"{npy_dir}/labs_percentiles_{split}.npy", mmap_mode='r')
        self.labs_miss = np.load(f"{npy_dir}/labs_missingness_{split}.npy", mmap_mode='r')
        
        # Extract labels
        self.labels = np.zeros((len(self.df), Config.NUM_LABELS), dtype=np.float32)
        for i, col in enumerate(Config.LABELS):
            if col in self.df.columns:
                values = self.df[col].fillna(0).values
                self.labels[:, i] = ((values == 1.0) | (values == -1.0)).astype(float)
        
        print(f"  ✅ {split.upper()}: {len(self):,} samples")
    
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
            np.concatenate([self.labs_pct[idx], self.labs_miss[idx]])
        ).float()
        
        labels = torch.from_numpy(self.labels[idx]).float()
        return cxr, ecg, labs, labels

# ===================== CELL 5: MODEL ARCHITECTURES =====================
# We need to define the architectures to load the weights

from torchvision import models

class EfficientNetB2MultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "EfficientNet-B2"
        self.feature_dim = 1408
        self.backbone = models.efficientnet_b2(weights=None)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1408, num_labels))
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features), features


class DenseNet121MultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "DenseNet-121"
        self.feature_dim = 1024
        self.backbone = models.densenet121(weights=None)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, num_labels))
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features), features


class ConvNeXtTinyMultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "ConvNeXt-Tiny"
        self.feature_dim = 768
        self.backbone = models.convnext_tiny(weights=None)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Flatten(1), nn.LayerNorm(768), nn.Dropout(0.3), nn.Linear(768, num_labels))
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.mean(dim=[-2, -1])
        return self.classifier(features), features


# Signal Models
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


class CNN1DMultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "1D-CNN"
        self.feature_dim = 256
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 11, padding=5),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 7, padding=3),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_labels))
    
    def forward(self, x):
        features = self.conv(x).squeeze(-1)
        return self.classifier(features), features


class ResNet1DMultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "ResNet-1D"
        self.feature_dim = 256
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, 15, stride=2, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
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
        return self.classifier(features), features


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


class InceptionTimeMultiLabel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "InceptionTime"
        self.feature_dim = 256
        self.inception1 = InceptionBlock(12, 32)
        self.inception2 = InceptionBlock(128, 32)
        self.inception3 = InceptionBlock(128, 32)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 256)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_labels))
    
    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.avgpool(x).squeeze(-1)
        features = self.fc(x)
        return self.classifier(features), features


# Clinical Models
class MLPMultiLabel(nn.Module):
    def __init__(self, input_dim=100, num_labels=6):
        super().__init__()
        self.name = "MLP"
        self.feature_dim = 64
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, num_labels))
    
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features), features


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
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.feature_dim, num_labels))
    
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
        return self.classifier(features), features


# ===================== CELL 6: LOAD PRETRAINED MODELS =====================
def load_best_model(modality, checkpoint_dir):
    """Load best model for a modality from checkpoint."""
    ckpt_path = f"{checkpoint_dir}/{modality}_best.pth"
    checkpoint = torch.load(ckpt_path, weights_only=False)
    
    model_name = checkpoint['model_name']
    feature_dim = checkpoint['feature_dim']
    
    print(f"  Loading {modality}: {model_name} (feat_dim={feature_dim})")
    
    # Create model based on name
    if modality == 'vision':
        if 'EfficientNet' in model_name:
            model = EfficientNetB2MultiLabel()
        elif 'DenseNet' in model_name:
            model = DenseNet121MultiLabel()
        else:
            model = ConvNeXtTinyMultiLabel()
    elif modality == 'signal':
        if 'InceptionTime' in model_name:
            model = InceptionTimeMultiLabel()
        elif 'ResNet' in model_name:
            model = ResNet1DMultiLabel()
        else:
            model = CNN1DMultiLabel()
    else:  # clinical
        if 'TabNet' in model_name:
            model = TabNetMultiLabel()
        else:
            model = MLPMultiLabel()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['best_auc']


print("\n📥 Loading pretrained models...")
vision_model, vision_auc = load_best_model('vision', CHECKPOINT_DIR)
signal_model, signal_auc = load_best_model('signal', CHECKPOINT_DIR)
clinical_model, clinical_auc = load_best_model('clinical', CHECKPOINT_DIR)

print(f"\n📊 Individual Model Performance:")
print(f"  Vision:   {vision_model.name:20s} | AUC: {vision_auc:.4f}")
print(f"  Signal:   {signal_model.name:20s} | AUC: {signal_auc:.4f}")
print(f"  Clinical: {clinical_model.name:20s} | AUC: {clinical_auc:.4f}")

# ===================== CELL 7: FUSION MODEL =====================
class MultiModalFusion(nn.Module):
    def __init__(self, vision_model, signal_model, clinical_model, num_labels=6, freeze_encoders=True):
        super().__init__()
        self.name = "MultiModal-Fusion"
        
        self.vision = vision_model
        self.signal = signal_model
        self.clinical = clinical_model
        
        # Optionally freeze pretrained encoders
        if freeze_encoders:
            for param in self.vision.parameters():
                param.requires_grad = False
            for param in self.signal.parameters():
                param.requires_grad = False
            for param in self.clinical.parameters():
                param.requires_grad = False
        
        # Feature dimensions
        v_dim = vision_model.feature_dim
        s_dim = signal_model.feature_dim
        c_dim = clinical_model.feature_dim
        total_dim = v_dim + s_dim + c_dim
        
        print(f"\n  🔀 Fusion Network:")
        print(f"     Vision ({vision_model.name}):     {v_dim}")
        print(f"     Signal ({signal_model.name}):     {s_dim}")
        print(f"     Clinical ({clinical_model.name}): {c_dim}")
        print(f"     Total:                     {total_dim}")
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )
    
    def forward(self, cxr, ecg, labs):
        # Extract features (no grad for frozen encoders)
        with torch.no_grad():
            _, v_feat = self.vision(cxr)
            _, s_feat = self.signal(ecg)
            _, c_feat = self.clinical(labs)
        
        # Concatenate and fuse
        combined = torch.cat([v_feat, s_feat, c_feat], dim=1)
        logits = self.fusion(combined)
        return logits, (v_feat, s_feat, c_feat)

# Create fusion model
fusion_model = MultiModalFusion(vision_model, signal_model, clinical_model, freeze_encoders=True)

# ===================== CELL 8: LOAD DATA =====================
print("\n📂 Loading datasets...")
train_dataset = MultiModalDataset(Config.DATA_DIR, 'train')
val_dataset = MultiModalDataset(Config.DATA_DIR, 'val')

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)

# ===================== CELL 9: TRAIN FUSION =====================
def compute_metrics(labels, probs):
    per_class_auc = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) < 2:
            per_class_auc.append(0.5)
        else:
            per_class_auc.append(roc_auc_score(labels[:, i], probs[:, i]))
    
    macro_auc = np.mean(per_class_auc)
    
    preds = (probs > 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    return {
        'macro_auc': macro_auc,
        'macro_f1': macro_f1,
        'per_class_auc': dict(zip(Config.LABELS, per_class_auc)),
        'labels': labels,
        'probs': probs
    }


fusion_model = fusion_model.to(DEVICE)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, fusion_model.parameters()),
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
scaler = GradScaler() if Config.USE_AMP else None

print("\n" + "="*70)
print("🔀 FUSION MODEL TRAINING")
print("="*70)

best_auc = 0
history = {'train_loss': [], 'val_auc': [], 'val_f1': []}
patience_counter = 0

for epoch in range(Config.EPOCHS):
    # Train
    fusion_model.train()
    total_loss = 0
    
    for cxr, ecg, labs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        cxr = cxr.to(DEVICE)
        ecg = ecg.to(DEVICE)
        labs = labs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        if Config.USE_AMP:
            with autocast():
                logits, _ = fusion_model(cxr, ecg, labs)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = fusion_model(cxr, ecg, labs)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    
    # Evaluate
    fusion_model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for cxr, ecg, labs, labels in val_loader:
            cxr = cxr.to(DEVICE)
            ecg = ecg.to(DEVICE)
            labs = labs.to(DEVICE)
            
            logits, _ = fusion_model(cxr, ecg, labs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_probs)
    
    history['train_loss'].append(avg_loss)
    history['val_auc'].append(metrics['macro_auc'])
    history['val_f1'].append(metrics['macro_f1'])
    
    marker = ""
    if metrics['macro_auc'] > best_auc:
        best_auc = metrics['macro_auc']
        best_metrics = metrics
        patience_counter = 0
        torch.save({
            'model_state_dict': fusion_model.state_dict(),
            'best_auc': best_auc,
            'epoch': epoch + 1,
            'metrics': {k: v for k, v in metrics.items() if k not in ['labels', 'probs']}
        }, f"{CHECKPOINT_DIR}/fusion_best.pth")
        marker = " ✅ Best!"
    else:
        patience_counter += 1
    
    print(f"Epoch {epoch+1:2d}/{Config.EPOCHS} | Loss: {avg_loss:.4f} | AUC: {metrics['macro_auc']:.4f} | F1: {metrics['macro_f1']:.4f}{marker}")
    
    if patience_counter >= 7:
        print("⏹️ Early stopping")
        break

# Load best model
checkpoint = torch.load(f"{CHECKPOINT_DIR}/fusion_best.pth", weights_only=False)
fusion_model.load_state_dict(checkpoint['model_state_dict'])

# ===================== CELL 10: FINAL RESULTS & VISUALIZATIONS =====================
print("\n" + "="*70)
print("🏆 FINAL RESULTS")
print("="*70)

print(f"\n📊 Model Comparison:")
print(f"  {'Model':<25} | {'Macro-AUC':>10}")
print(f"  {'-'*25}-+-{'-'*10}")
print(f"  {f'Vision ({vision_model.name})':<25} | {vision_auc:>10.4f}")
print(f"  {f'Signal ({signal_model.name})':<25} | {signal_auc:>10.4f}")
print(f"  {f'Clinical ({clinical_model.name})':<25} | {clinical_auc:>10.4f}")
print(f"  {'-'*25}-+-{'-'*10}")
print(f"  {'🔀 FUSION':<25} | {best_auc:>10.4f} 🏆")

improvement = best_auc - max(vision_auc, signal_auc, clinical_auc)
print(f"\n  📈 Fusion Improvement: +{improvement:.4f} over best single modality")

print(f"\n📊 Per-Class AUC (Fusion):")
for label, auc in best_metrics['per_class_auc'].items():
    bar = "█" * int(auc * 20) + "░" * (20 - int(auc * 20))
    print(f"  {label:<20} {bar} {auc:.4f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Model Comparison Bar Chart
ax1 = axes[0, 0]
models = [f'Vision\n({vision_model.name})', f'Signal\n({signal_model.name})', f'Clinical\n({clinical_model.name})', 'FUSION']
aucs = [vision_auc, signal_auc, clinical_auc, best_auc]
colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
bars = ax1.bar(models, aucs, color=colors, edgecolor='black', linewidth=2)
ax1.set_ylabel('Macro-AUC', fontsize=12)
ax1.set_title('🏆 Multi-Modal Fusion vs Single Modalities', fontsize=14, fontweight='bold')
ax1.set_ylim(0.4, 1.0)
ax1.axhline(y=best_auc, color='green', linestyle='--', alpha=0.5, label=f'Fusion: {best_auc:.4f}')
for bar, auc in zip(bars, aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auc:.4f}', ha='center', fontweight='bold', fontsize=11)
ax1.legend()

# 2. Per-Class AUC
ax2 = axes[0, 1]
class_names = list(best_metrics['per_class_auc'].keys())
class_aucs = list(best_metrics['per_class_auc'].values())
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(class_names)))
bars = ax2.barh(class_names, class_aucs, color=colors, edgecolor='black')
ax2.set_xlabel('AUC-ROC')
ax2.set_title('📊 Per-Disease AUC (Fusion Model)', fontsize=14, fontweight='bold')
ax2.set_xlim(0.4, 1.0)
ax2.axvline(x=best_auc, color='red', linestyle='--', label=f'Macro: {best_auc:.4f}')
for bar, auc in zip(bars, class_aucs):
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{auc:.3f}', va='center')
ax2.legend()

# 3. Training History
ax3 = axes[1, 0]
ax3.plot(history['val_auc'], 'g-o', label='Validation AUC', linewidth=2, markersize=6)
ax3.plot(history['val_f1'], 'b-s', label='Validation F1', linewidth=2, markersize=6)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Score')
ax3.set_title('📈 Fusion Training History', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. ROC Curves
ax4 = axes[1, 1]
for i, label in enumerate(Config.LABELS):
    fpr, tpr, _ = roc_curve(best_metrics['labels'][:, i], best_metrics['probs'][:, i])
    ax4.plot(fpr, tpr, label=f'{label} ({best_metrics["per_class_auc"][label]:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('📉 ROC Curves (All Diseases)', fontsize=14, fontweight='bold')
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fusion_final_results.png", dpi=300, bbox_inches='tight')
print(f"\n📊 Saved: {OUTPUT_DIR}/fusion_final_results.png")
plt.show()

# ===================== CELL 11: SAVE FINAL REPORT =====================
report = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': 'SYMILE-MIMIC',
    'num_labels': Config.NUM_LABELS,
    'labels': Config.LABELS,
    'individual_models': {
        'vision': {'model': vision_model.name, 'auc': vision_auc},
        'signal': {'model': signal_model.name, 'auc': signal_auc},
        'clinical': {'model': clinical_model.name, 'auc': clinical_auc}
    },
    'fusion': {
        'auc': best_auc,
        'f1': best_metrics['macro_f1'],
        'improvement': improvement,
        'per_class_auc': best_metrics['per_class_auc']
    },
    'training': {
        'epochs': len(history['val_auc']),
        'batch_size': Config.BATCH_SIZE,
        'learning_rate': Config.LEARNING_RATE
    }
}

with open(f"{OUTPUT_DIR}/final_report.json", 'w') as f:
    json.dump(report, f, indent=2)

# Text report
with open(f"{OUTPUT_DIR}/final_report.txt", 'w') as f:
    f.write("="*70 + "\n")
    f.write("VisionCare - Multi-Modal Multi-Label Training Report\n")
    f.write("="*70 + "\n\n")
    f.write(f"Dataset: SYMILE-MIMIC\n")
    f.write(f"Date: {report['timestamp']}\n")
    f.write(f"Labels: {Config.NUM_LABELS} diseases\n\n")
    
    f.write("INDIVIDUAL MODEL RESULTS\n")
    f.write("-"*40 + "\n")
    f.write(f"Vision ({vision_model.name}):   AUC = {vision_auc:.4f}\n")
    f.write(f"Signal ({signal_model.name}):   AUC = {signal_auc:.4f}\n")
    f.write(f"Clinical ({clinical_model.name}): AUC = {clinical_auc:.4f}\n\n")
    
    f.write("FUSION MODEL RESULTS\n")
    f.write("-"*40 + "\n")
    f.write(f"Macro-AUC:   {best_auc:.4f}\n")
    f.write(f"Macro-F1:    {best_metrics['macro_f1']:.4f}\n")
    f.write(f"Improvement: +{improvement:.4f}\n\n")
    
    f.write("PER-CLASS AUC\n")
    f.write("-"*40 + "\n")
    for label, auc in best_metrics['per_class_auc'].items():
        f.write(f"{label:<20}: {auc:.4f}\n")

print(f"📄 Saved: {OUTPUT_DIR}/final_report.json")
print(f"📄 Saved: {OUTPUT_DIR}/final_report.txt")

print("\n" + "="*70)
print("✅ ALL TRAINING COMPLETE!")
print("="*70)
print(f"\n🎉 Final Fusion AUC: {best_auc:.4f}")
print(f"📁 All results saved to: {OUTPUT_DIR}/")
