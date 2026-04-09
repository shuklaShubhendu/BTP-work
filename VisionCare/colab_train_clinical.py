"""
VisionCare - CLINICAL MODULE Training (Labs - Blood Tests)
===========================================================
Trains 2 models and saves the BEST to shared Google Drive.

Models: MLP, TabNet

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

# ===================== CELL 3: CONFIG =====================
class Config:
    DATA_DIR = SHARED_FOLDER
    LABELS = ['Cardiomegaly', 'Edema', 'Atelectasis', 'Pleural Effusion', 'Lung Opacity', 'No Finding']
    NUM_LABELS = 6
    BATCH_SIZE = 128  # Labs data is small, can use large batch
    NUM_WORKERS = 0
    EPOCHS = 30  # More epochs for small model
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    USE_AMP = False  # Not needed for small models

# ===================== CELL 4: DATASET =====================
class ClinicalDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.split = split
        csv_path = f"{data_dir}/{split}.csv"
        npy_dir = f"{data_dir}/data_npy/{split}"
        
        print(f"  📂 Loading {split} data...")
        self.df = pd.read_csv(csv_path)
        
        # Labs: 50 percentiles + 50 missingness = 100 features
        self.labs_pct = np.load(f"{npy_dir}/labs_percentiles_{split}.npy", mmap_mode='r')
        self.labs_miss = np.load(f"{npy_dir}/labs_missingness_{split}.npy", mmap_mode='r')
        
        # Extract labels
        self.labels = np.zeros((len(self.df), Config.NUM_LABELS), dtype=np.float32)
        for i, col in enumerate(Config.LABELS):
            if col in self.df.columns:
                values = self.df[col].fillna(0).values
                self.labels[:, i] = ((values == 1.0) | (values == -1.0)).astype(float)
        
        print(f"  ✅ {split.upper()}: {len(self):,} samples, Labs: (100,)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        labs = torch.from_numpy(
            np.concatenate([self.labs_pct[idx], self.labs_miss[idx]])
        ).float()
        labels = torch.from_numpy(self.labels[idx]).float()
        return labs, labels

# ===================== CELL 5: MODELS =====================

class MLPMultiLabel(nn.Module):
    """MLP for blood lab values."""
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
    """TabNet - Attention-based feature selection for tabular data."""
    def __init__(self, input_dim=100, n_steps=3, hidden_dim=64, num_labels=6):
        super().__init__()
        self.name = "TabNet"
        self.feature_dim = 64
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

# ===================== CELL 6: TRAINING FUNCTIONS =====================
def compute_metrics(labels, probs):
    per_class_auc = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) < 2:
            per_class_auc.append(0.5)
        else:
            per_class_auc.append(roc_auc_score(labels[:, i], probs[:, i]))
    return {'macro_auc': np.mean(per_class_auc)}


def train_model(model, train_loader, val_loader, epochs=30, save_name='model'):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_auc = 0
    history = {'train_loss': [], 'val_auc': []}
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"🚀 Training: {model.name}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for labs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            labs, labels = labs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            logits, _ = model(labs)
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
            for labs, labels in val_loader:
                labs = labs.to(DEVICE)
                logits, _ = model(labs)
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
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_auc': best_auc,
                'epoch': epoch + 1,
                'model_name': model.name,
                'feature_dim': model.feature_dim
            }, f"{CHECKPOINT_DIR}/{save_name}.pth")
            marker = " ✅ Best!"
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Macro-AUC: {metrics['macro_auc']:.4f}{marker}")
        
        if patience_counter >= 10:
            print("⏹️ Early stopping")
            break
    
    checkpoint = torch.load(f"{CHECKPOINT_DIR}/{save_name}.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return {'model_name': model.name, 'best_auc': best_auc, 'history': history, 'feature_dim': model.feature_dim}

# ===================== CELL 7: LOAD DATA =====================
print("\n📂 Loading datasets...")
train_dataset = ClinicalDataset(Config.DATA_DIR, 'train')
val_dataset = ClinicalDataset(Config.DATA_DIR, 'val')

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)

# ===================== CELL 8: TRAIN BOTH MODELS =====================
print("\n" + "="*70)
print("🧪 CLINICAL MODULE - Comparing 2 Models")
print("="*70)

results = []

# Model 1: MLP
model1 = MLPMultiLabel()
result1 = train_model(model1, train_loader, val_loader, Config.EPOCHS, 'clinical_mlp')
results.append(result1)
torch.cuda.empty_cache()

# Model 2: TabNet
model2 = TabNetMultiLabel()
result2 = train_model(model2, train_loader, val_loader, Config.EPOCHS, 'clinical_tabnet')
results.append(result2)
torch.cuda.empty_cache()

# ===================== CELL 9: FIND BEST & SAVE =====================
print("\n" + "="*70)
print("📊 CLINICAL MODEL COMPARISON")
print("="*70)

best_result = max(results, key=lambda x: x['best_auc'])
for r in results:
    marker = " 🏆 BEST" if r == best_result else ""
    print(f"  {r['model_name']:20s} | Macro-AUC: {r['best_auc']:.4f}{marker}")

# Copy best model
import shutil
name_map = {'MLP': 'mlp', 'TabNet': 'tabnet'}
shutil.copy(f"{CHECKPOINT_DIR}/clinical_{name_map[best_result['model_name']]}.pth", f"{CHECKPOINT_DIR}/clinical_best.pth")

# Save comparison
with open(f"{OUTPUT_DIR}/clinical_comparison.json", 'w') as f:
    json.dump({
        'modality': 'clinical',
        'best_model': best_result['model_name'],
        'best_auc': best_result['best_auc'],
        'best_feature_dim': best_result['feature_dim'],
        'all_results': [{'name': r['model_name'], 'auc': r['best_auc']} for r in results]
    }, f, indent=2)

print(f"\n✅ Best Clinical Model: {best_result['model_name']} (AUC: {best_result['best_auc']:.4f})")
print(f"💾 Saved to: {CHECKPOINT_DIR}/clinical_best.pth")

# ===================== CELL 10: PLOT RESULTS =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
names = [r['model_name'] for r in results]
aucs = [r['best_auc'] for r in results]
colors = ['#2ecc71' if r == best_result else '#9b59b6' for r in results]
bars = ax1.bar(names, aucs, color=colors, edgecolor='black')
ax1.set_ylabel('Macro-AUC')
ax1.set_title('🧪 Clinical Model Comparison')
ax1.set_ylim(0.4, 0.8)
for bar, auc in zip(bars, aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auc:.4f}', ha='center', fontweight='bold')

ax2 = axes[1]
ax2.plot(best_result['history']['val_auc'], 'm-o', label='Validation AUC')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Macro-AUC')
ax2.set_title(f"📈 {best_result['model_name']} Training History")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/clinical_results.png", dpi=300, bbox_inches='tight')
print(f"📊 Plot saved to: {OUTPUT_DIR}/clinical_results.png")
plt.show()

print("\n" + "="*70)
print("✅ CLINICAL MODULE COMPLETE!")
print("="*70)
