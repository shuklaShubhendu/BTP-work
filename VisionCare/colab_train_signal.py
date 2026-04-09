"""
VisionCare - SIGNAL MODULE Training (ECG - 12-Lead)
====================================================
Trains 3 models and saves the BEST to shared Google Drive.

Models: 1D-CNN, ResNet-1D, InceptionTime

USAGE: Copy to Colab notebook, run all cells.
OUTPUT: Best model saved to shared Google Drive folder.
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
print(f"💾 Output: {OUTPUT_DIR}")

# ===================== CELL 2: IMPORTS =====================
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import roc_auc_score

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
    BATCH_SIZE = 64  # ECG is smaller, can use larger batch
    NUM_WORKERS = 0
    EPOCHS = 15
    LEARNING_RATE = 1e-3  # Slightly higher LR for signal models
    WEIGHT_DECAY = 1e-5
    USE_AMP = True

# ===================== CELL 4: DATASET =====================
class SignalDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.split = split
        csv_path = f"{data_dir}/{split}.csv"
        npy_dir = f"{data_dir}/data_npy/{split}"
        
        print(f"  📂 Loading {split} data...")
        self.df = pd.read_csv(csv_path)
        self.ecg = np.load(f"{npy_dir}/ecg_{split}.npy", mmap_mode='r')
        
        # Extract labels
        self.labels = np.zeros((len(self.df), Config.NUM_LABELS), dtype=np.float32)
        for i, col in enumerate(Config.LABELS):
            if col in self.df.columns:
                values = self.df[col].fillna(0).values
                self.labels[:, i] = ((values == 1.0) | (values == -1.0)).astype(float)
        
        print(f"  ✅ {split.upper()}: {len(self):,} samples, ECG shape: {self.ecg.shape}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        ecg = torch.from_numpy(self.ecg[idx].copy()).float()
        # Reshape: (1, 5000, 12) -> (12, 5000)
        if ecg.dim() == 3:
            ecg = ecg.squeeze(0)
        if ecg.shape[0] != 12:
            ecg = ecg.transpose(0, 1)
        labels = torch.from_numpy(self.labels[idx]).float()
        return ecg, labels

# ===================== CELL 5: MODELS =====================

class CNN1DMultiLabel(nn.Module):
    """1D CNN for ECG - Fast baseline."""
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
    """ResNet-1D for ECG - Deep residual network."""
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
    """InceptionTime - SOTA for time-series classification."""
    def __init__(self, num_labels=6):
        super().__init__()
        self.name = "InceptionTime"
        self.feature_dim = 256
        
        self.inception1 = InceptionBlock(12, 32)   # -> 128 channels
        self.inception2 = InceptionBlock(128, 32)  # -> 128 channels
        self.inception3 = InceptionBlock(128, 32)  # -> 128 channels
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

# ===================== CELL 6: TRAINING FUNCTIONS =====================
def compute_metrics(labels, probs):
    per_class_auc = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) < 2:
            per_class_auc.append(0.5)
        else:
            per_class_auc.append(roc_auc_score(labels[:, i], probs[:, i]))
    return {'macro_auc': np.mean(per_class_auc), 'per_class_auc': dict(zip(Config.LABELS, per_class_auc))}


def train_model(model, train_loader, val_loader, epochs=15, save_name='model'):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler() if Config.USE_AMP else None
    
    best_auc = 0
    history = {'train_loss': [], 'val_auc': []}
    
    print(f"\n{'='*60}")
    print(f"🚀 Training: {model.name}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for ecg, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            ecg, labels = ecg.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            if Config.USE_AMP:
                with autocast():
                    logits, _ = model(ecg)
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(ecg)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for ecg, labels in val_loader:
                ecg = ecg.to(DEVICE)
                logits, _ = model(ecg)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_labels, all_probs)
        
        history['train_loss'].append(avg_loss)
        history['val_auc'].append(metrics['macro_auc'])
        
        marker = ""
        if metrics['macro_auc'] > best_auc:
            best_auc = metrics['macro_auc']
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_auc': best_auc,
                'epoch': epoch + 1,
                'model_name': model.name,
                'feature_dim': model.feature_dim
            }, f"{CHECKPOINT_DIR}/{save_name}.pth")
            marker = " ✅ Best!"
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Macro-AUC: {metrics['macro_auc']:.4f}{marker}")
        
        if len(history['val_auc']) > 5 and max(history['val_auc'][-5:]) <= best_auc - 0.01:
            print("⏹️ Early stopping")
            break
    
    checkpoint = torch.load(f"{CHECKPOINT_DIR}/{save_name}.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return {'model_name': model.name, 'best_auc': best_auc, 'history': history, 'feature_dim': model.feature_dim}

# ===================== CELL 7: LOAD DATA =====================
print("\n📂 Loading datasets...")
train_dataset = SignalDataset(Config.DATA_DIR, 'train')
val_dataset = SignalDataset(Config.DATA_DIR, 'val')

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)

# ===================== CELL 8: TRAIN ALL 3 MODELS =====================
print("\n" + "="*70)
print("❤️ SIGNAL MODULE - Comparing 3 Models")
print("="*70)

results = []

# Model 1: 1D-CNN
model1 = CNN1DMultiLabel()
result1 = train_model(model1, train_loader, val_loader, Config.EPOCHS, 'signal_cnn1d')
results.append(result1)
torch.cuda.empty_cache()

# Model 2: ResNet-1D
model2 = ResNet1DMultiLabel()
result2 = train_model(model2, train_loader, val_loader, Config.EPOCHS, 'signal_resnet1d')
results.append(result2)
torch.cuda.empty_cache()

# Model 3: InceptionTime
model3 = InceptionTimeMultiLabel()
result3 = train_model(model3, train_loader, val_loader, Config.EPOCHS, 'signal_inceptiontime')
results.append(result3)
torch.cuda.empty_cache()

# ===================== CELL 9: FIND BEST & SAVE =====================
print("\n" + "="*70)
print("📊 SIGNAL MODEL COMPARISON")
print("="*70)

best_result = max(results, key=lambda x: x['best_auc'])
for r in results:
    marker = " 🏆 BEST" if r == best_result else ""
    print(f"  {r['model_name']:20s} | Macro-AUC: {r['best_auc']:.4f}{marker}")

# Copy best model
import shutil
name_map = {'1D-CNN': 'cnn1d', 'ResNet-1D': 'resnet1d', 'InceptionTime': 'inceptiontime'}
shutil.copy(f"{CHECKPOINT_DIR}/signal_{name_map[best_result['model_name']]}.pth", f"{CHECKPOINT_DIR}/signal_best.pth")

# Save comparison
with open(f"{OUTPUT_DIR}/signal_comparison.json", 'w') as f:
    json.dump({
        'modality': 'signal',
        'best_model': best_result['model_name'],
        'best_auc': best_result['best_auc'],
        'best_feature_dim': best_result['feature_dim'],
        'all_results': [{'name': r['model_name'], 'auc': r['best_auc']} for r in results]
    }, f, indent=2)

print(f"\n✅ Best Signal Model: {best_result['model_name']} (AUC: {best_result['best_auc']:.4f})")
print(f"💾 Saved to: {CHECKPOINT_DIR}/signal_best.pth")

# ===================== CELL 10: PLOT RESULTS =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
names = [r['model_name'] for r in results]
aucs = [r['best_auc'] for r in results]
colors = ['#2ecc71' if r == best_result else '#e74c3c' for r in results]
bars = ax1.bar(names, aucs, color=colors, edgecolor='black')
ax1.set_ylabel('Macro-AUC')
ax1.set_title('❤️ Signal Model Comparison')
ax1.set_ylim(0.4, 0.8)
for bar, auc in zip(bars, aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auc:.4f}', ha='center', fontweight='bold')

ax2 = axes[1]
ax2.plot(best_result['history']['val_auc'], 'r-o', label='Validation AUC')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Macro-AUC')
ax2.set_title(f"📈 {best_result['model_name']} Training History")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/signal_results.png", dpi=300, bbox_inches='tight')
print(f"📊 Plot saved to: {OUTPUT_DIR}/signal_results.png")
plt.show()

print("\n" + "="*70)
print("✅ SIGNAL MODULE COMPLETE!")
print("="*70)
