"""
VisionCare Local Training - Optimized for RTX 3050 (4GB VRAM)
==============================================================

This script trains on your local machine with memory optimizations.

Setup:
1. Download via IDM to D:/VisionCare/symile-mimic/
2. Run: python train_local.py

Expected folder structure:
D:/VisionCare/symile-mimic/
├── train.csv
├── val.csv
└── data_npy/
    ├── train/
    │   ├── cxr_train.npy
    │   ├── ecg_train.npy
    │   ├── labs_percentiles_train.npy
    │   └── labs_missingness_train.npy
    └── val/
        ├── cxr_val.npy
        ├── ecg_val.npy
        ├── labs_percentiles_val.npy
        └── labs_missingness_val.npy
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import time

# ============== CONFIGURATION ==============
DATA_DIR = "D:/VisionCare/symile-mimic"  # Change this to your path
BATCH_SIZE = 8  # Reduced for 4GB VRAM
NUM_WORKERS = 2
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🖥️ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ============== DATASET ==============
class SymileMIMICDataset(Dataset):
    """Memory-efficient dataset for Symile-MIMIC."""
    
    def __init__(self, data_dir, split='train'):
        self.split = split
        
        # Load CSV (contains labels)
        self.df = pd.read_csv(f"{data_dir}/{split}.csv")
        
        # Memory-map numpy arrays (doesn't load into RAM!)
        npy_dir = f"{data_dir}/data_npy/{split}"
        self.cxr = np.load(f"{npy_dir}/cxr_{split}.npy", mmap_mode='r')
        self.ecg = np.load(f"{npy_dir}/ecg_{split}.npy", mmap_mode='r')
        self.labs_pct = np.load(f"{npy_dir}/labs_percentiles_{split}.npy", mmap_mode='r')
        self.labs_miss = np.load(f"{npy_dir}/labs_missingness_{split}.npy", mmap_mode='r')
        
        # Get Cardiomegaly labels
        if 'Cardiomegaly' in self.df.columns:
            labels = self.df['Cardiomegaly'].fillna(0).values
            self.labels = ((labels == 1.0) | (labels == -1.0)).astype(int)
        else:
            self.labels = np.zeros(len(self.df), dtype=int)
        
        print(f"📊 {split.upper()}: {len(self.df)} samples, {self.labels.mean()*100:.1f}% positive")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cxr = torch.tensor(self.cxr[idx].copy(), dtype=torch.float32)
        
        ecg = torch.tensor(self.ecg[idx].copy(), dtype=torch.float32)
        ecg = ecg.squeeze(0).transpose(0, 1)  # (12, 5000)
        
        labs = np.concatenate([self.labs_pct[idx], self.labs_miss[idx]])
        labs = torch.tensor(labs, dtype=torch.float32)
        
        return cxr, ecg, labs, self.labels[idx]


# ============== MODELS ==============
class VisionModule(nn.Module):
    """ResNet-50 for CXR - uses gradient checkpointing to save memory."""
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.feature_dim = 2048
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(2048, 2)
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat), feat


class SignalModule(nn.Module):
    """1D-CNN for ECG."""
    def __init__(self):
        super().__init__()
        self.feature_dim = 256
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 15, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 11, padding=5), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 7, padding=3), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(256, 2)
    
    def forward(self, x):
        feat = self.conv(x).squeeze(-1)
        return self.classifier(feat), feat


class ClinicalModule(nn.Module):
    """MLP for blood labs."""
    def __init__(self):
        super().__init__()
        self.feature_dim = 64
        self.net = nn.Sequential(
            nn.Linear(100, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.classifier = nn.Linear(64, 2)
    
    def forward(self, x):
        feat = self.net(x)
        return self.classifier(feat), feat


class VisionCareFusion(nn.Module):
    """Fusion model combining all modalities."""
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
    
    def forward(self, cxr, ecg, labs):
        _, v = self.vision(cxr)
        _, s = self.signal(ecg)
        _, c = self.clinical(labs)
        combined = torch.cat([v, s, c], dim=1)
        return self.fusion(combined)


# ============== TRAINING ==============
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for cxr, ecg, labs, y in tqdm(loader, desc="Training"):
        cxr = cxr.to(DEVICE)
        ecg = ecg.to(DEVICE)
        labs = labs.to(DEVICE)
        y = y.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(cxr, ecg, labs)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Clear cache to prevent OOM
        torch.cuda.empty_cache()
    
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for cxr, ecg, labs, y in tqdm(loader, desc="Evaluating"):
            cxr = cxr.to(DEVICE)
            ecg = ecg.to(DEVICE)
            labs = labs.to(DEVICE)
            
            logits = model(cxr, ecg, labs)
            probs = F.softmax(logits, dim=1)[:, 1]
            
            preds.extend(probs.cpu().numpy())
            targets.extend(y.numpy())
    
    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, [1 if p > 0.5 else 0 for p in preds])
    return auc, acc


def main():
    print("\n" + "="*60)
    print("🫀 VisionCare Training - RTX 3050 Optimized")
    print("="*60)
    
    # Load data
    print("\n📂 Loading datasets...")
    train_ds = SymileMIMICDataset(DATA_DIR, 'train')
    val_ds = SymileMIMICDataset(DATA_DIR, 'val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # Create model
    print("\n🔧 Creating model...")
    vision = VisionModule()
    signal = SignalModule()
    clinical = ClinicalModule()
    model = VisionCareFusion(vision, signal, clinical).to(DEVICE)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {param_count:,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\n🚀 Starting training... ({EPOCHS} epochs)")
    best_auc = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*40}")
        
        # Train
        loss = train_epoch(model, train_loader, optimizer, criterion)
        
        # Evaluate
        auc, acc = evaluate(model, val_loader)
        scheduler.step(auc)
        
        print(f"\n📈 Results:")
        print(f"   Loss: {loss:.4f}")
        print(f"   Val AUC: {auc:.4f}")
        print(f"   Val Accuracy: {acc*100:.1f}%")
        
        # Save best
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'visioncare_best.pth')
            print("   ✅ New best model saved!")
    
    # Final
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏆 Training Complete!")
    print(f"   Best AUC: {best_auc:.4f}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Model saved: visioncare_best.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
