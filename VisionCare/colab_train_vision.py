"""
VisionCare - VISION MODULE Training (CXR - Chest X-Ray)
========================================================
Trains 3 models and saves the BEST to shared Google Drive.

Models: DenseNet-121, EfficientNet-B2, ConvNeXt-Tiny

USAGE: Copy to Colab notebook, run all cells.
OUTPUT: Best model saved to shared Google Drive folder.
"""

# ===================== CELL 1: SETUP & MOUNT DRIVE =====================
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ========== AUTO-DETECT DATA FOLDER ==========
# Try multiple possible paths for the shared folder
POSSIBLE_PATHS = [
    "/content/drive/MyDrive/symile-mimic",
    "/content/drive/My Drive/symile-mimic", 
    "/content/drive/Shareddrives/symile-mimic",
    "/content/drive/MyDrive/Shared with me/symile-mimic",
    "/content/drive/MyDrive/symile_mimic",
]

SHARED_FOLDER = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path) and os.path.exists(f"{path}/train.csv"):
        SHARED_FOLDER = path
        break

# If not found, list what's in drive to help debug
if SHARED_FOLDER is None:
    print("❌ Data folder NOT found! Looking for alternatives...")
    
    # Check what folders exist in MyDrive
    mydrive = "/content/drive/MyDrive"
    if os.path.exists(mydrive):
        folders = [f for f in os.listdir(mydrive) if 'symile' in f.lower() or 'mimic' in f.lower()]
        if folders:
            print(f"   Found similar folders: {folders}")
            # Try the first match
            for folder in folders:
                test_path = f"{mydrive}/{folder}"
                if os.path.exists(f"{test_path}/train.csv"):
                    SHARED_FOLDER = test_path
                    break
    
    if SHARED_FOLDER is None:
        print("\n⚠️ MANUAL SETUP REQUIRED:")
        print("   1. Go to: https://drive.google.com/drive/folders/12Ris_8Kav_MQoO5Luxw3Ylq33m-ewX_p")
        print("   2. Right-click the 'symile-mimic' folder")
        print("   3. Click 'Organize' -> 'Add shortcut' -> 'My Drive'")
        print("   4. Re-run this cell")
        raise FileNotFoundError("Data folder not found. See instructions above.")

OUTPUT_DIR = f"{SHARED_FOLDER}/MultiLabel_Results"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"✅ Data folder found: {SHARED_FOLDER}")
print(f"💾 Output folder: {OUTPUT_DIR}")

# ===================== CELL 2: IMPORTS =====================
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import models

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

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
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    USE_AMP = True
    LABEL_SMOOTHING = 0.1

# ===================== CELL 4: DATASET =====================
class VisionDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.split = split
        csv_path = f"{data_dir}/{split}.csv"
        npy_dir = f"{data_dir}/data_npy/{split}"
        
        print(f"  📂 Loading {split} data...")
        self.df = pd.read_csv(csv_path)
        self.cxr = np.load(f"{npy_dir}/cxr_{split}.npy", mmap_mode='r')
        
        # Extract labels
        self.labels = np.zeros((len(self.df), Config.NUM_LABELS), dtype=np.float32)
        for i, col in enumerate(Config.LABELS):
            if col in self.df.columns:
                values = self.df[col].fillna(0).values
                self.labels[:, i] = ((values == 1.0) | (values == -1.0)).astype(float)
        
        print(f"  ✅ {split.upper()}: {len(self):,} samples, CXR shape: {self.cxr.shape}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cxr = torch.from_numpy(self.cxr[idx].copy()).float()
        labels = torch.from_numpy(self.labels[idx]).float()
        return cxr, labels

# ===================== CELL 5: MODELS =====================
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
        if features.dim() > 2:
            features = features.mean(dim=[-2, -1])
        return self.classifier(features), features

# ===================== CELL 6: TRAINING FUNCTIONS =====================
def compute_metrics(labels, probs):
    preds = (probs > 0.5).astype(int)
    per_class_auc = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) < 2:
            per_class_auc.append(0.5)
        else:
            per_class_auc.append(roc_auc_score(labels[:, i], probs[:, i]))
    return {
        'macro_auc': np.mean(per_class_auc),
        'per_class_auc': dict(zip(Config.LABELS, per_class_auc)),
        'labels': labels,
        'probs': probs
    }


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
        # Train
        model.train()
        total_loss = 0
        for cxr, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            cxr, labels = cxr.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            if Config.USE_AMP:
                with autocast():
                    logits, _ = model(cxr)
                    # Label smoothing
                    smooth_labels = labels * 0.9 + 0.05
                    loss = F.binary_cross_entropy_with_logits(logits, smooth_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(cxr)
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
            for cxr, labels in val_loader:
                cxr = cxr.to(DEVICE)
                logits, _ = model(cxr)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_labels, all_probs)
        
        history['train_loss'].append(avg_loss)
        history['val_auc'].append(metrics['macro_auc'])
        
        # Save best
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
        
        # Early stopping
        if len(history['val_auc']) > 5 and max(history['val_auc'][-5:]) <= best_auc - 0.01:
            print("⏹️ Early stopping")
            break
    
    # Load best model
    checkpoint = torch.load(f"{CHECKPOINT_DIR}/{save_name}.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return {'model_name': model.name, 'best_auc': best_auc, 'history': history, 'feature_dim': model.feature_dim}

# ===================== CELL 7: LOAD DATA =====================
print("\n📂 Loading datasets...")
train_dataset = VisionDataset(Config.DATA_DIR, 'train')
val_dataset = VisionDataset(Config.DATA_DIR, 'val')

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)

# ===================== CELL 8: TRAIN ALL 3 MODELS =====================
print("\n" + "="*70)
print("🩻 VISION MODULE - Comparing 3 Models")
print("="*70)

results = []

# Model 1: DenseNet-121
model1 = DenseNet121MultiLabel()
result1 = train_model(model1, train_loader, val_loader, Config.EPOCHS, 'vision_densenet')
results.append(result1)
torch.cuda.empty_cache()

# Model 2: EfficientNet-B2
model2 = EfficientNetB2MultiLabel()
result2 = train_model(model2, train_loader, val_loader, Config.EPOCHS, 'vision_efficientnet')
results.append(result2)
torch.cuda.empty_cache()

# Model 3: ConvNeXt-Tiny
model3 = ConvNeXtTinyMultiLabel()
result3 = train_model(model3, train_loader, val_loader, Config.EPOCHS, 'vision_convnext')
results.append(result3)
torch.cuda.empty_cache()

# ===================== CELL 9: FIND BEST & SAVE =====================
print("\n" + "="*70)
print("📊 VISION MODEL COMPARISON")
print("="*70)

best_result = max(results, key=lambda x: x['best_auc'])
for r in results:
    marker = " 🏆 BEST" if r == best_result else ""
    print(f"  {r['model_name']:20s} | Macro-AUC: {r['best_auc']:.4f}{marker}")

# Copy best model as the main vision checkpoint
import shutil
best_name = best_result['model_name'].lower().replace('-', '').replace(' ', '')
shutil.copy(f"{CHECKPOINT_DIR}/vision_{best_name}.pth" if 'densenet' in best_name else 
            f"{CHECKPOINT_DIR}/vision_{'densenet' if best_result['model_name']=='DenseNet-121' else 'efficientnet' if best_result['model_name']=='EfficientNet-B2' else 'convnext'}.pth",
            f"{CHECKPOINT_DIR}/vision_best.pth")

# Save comparison results
with open(f"{OUTPUT_DIR}/vision_comparison.json", 'w') as f:
    json.dump({
        'modality': 'vision',
        'best_model': best_result['model_name'],
        'best_auc': best_result['best_auc'],
        'best_feature_dim': best_result['feature_dim'],
        'all_results': [{'name': r['model_name'], 'auc': r['best_auc']} for r in results]
    }, f, indent=2)

print(f"\n✅ Best Vision Model: {best_result['model_name']} (AUC: {best_result['best_auc']:.4f})")
print(f"💾 Saved to: {CHECKPOINT_DIR}/vision_best.pth")
print(f"📄 Comparison saved to: {OUTPUT_DIR}/vision_comparison.json")

# ===================== CELL 10: PLOT RESULTS =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Model comparison bar chart
ax1 = axes[0]
names = [r['model_name'] for r in results]
aucs = [r['best_auc'] for r in results]
colors = ['#2ecc71' if r == best_result else '#3498db' for r in results]
bars = ax1.bar(names, aucs, color=colors, edgecolor='black')
ax1.set_ylabel('Macro-AUC')
ax1.set_title('🩻 Vision Model Comparison')
ax1.set_ylim(0.5, 1.0)
for bar, auc in zip(bars, aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auc:.4f}', ha='center', fontweight='bold')

# Training history of best model
ax2 = axes[1]
ax2.plot(best_result['history']['val_auc'], 'b-o', label='Validation AUC')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Macro-AUC')
ax2.set_title(f"📈 {best_result['model_name']} Training History")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/vision_results.png", dpi=300, bbox_inches='tight')
print(f"📊 Plot saved to: {OUTPUT_DIR}/vision_results.png")
plt.show()

print("\n" + "="*70)
print("✅ VISION MODULE COMPLETE!")
print("="*70)
