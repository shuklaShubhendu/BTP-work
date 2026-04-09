"""
VisionCare - Symile-MIMIC Data Loading Script
==============================================
Copy this code to your Colab notebook!

Data Structure in data_npy/:
- cxr_*.npy: (n, 3, 320, 320) - CXR images, ImageNet normalized
- ecg_*.npy: (n, 1, 5000, 12) - 12-lead ECG, 10 sec @ 500Hz
- labs_percentiles_*.npy: (n, 50) - Lab values as percentiles
- labs_missingness_*.npy: (n, 50) - Missing lab indicators

Labels: CheXpert Cardiomegaly from CSV files!
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SymileMIMICDataset(Dataset):
    """
    Load Symile-MIMIC preprocessed data with REAL labels.
    
    Uses CheXpert Cardiomegaly label for CVD classification!
    """
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        
        # Load metadata CSV (contains CheXpert labels!)
        self.df = pd.read_csv(f"{data_dir}/{split}.csv")
        
        # Load numpy arrays (memory-mapped for efficiency)
        npy_dir = f"{data_dir}/data_npy/{split}"
        self.cxr = np.load(f"{npy_dir}/cxr_{split}.npy", mmap_mode='r')
        self.ecg = np.load(f"{npy_dir}/ecg_{split}.npy", mmap_mode='r')
        self.labs_pct = np.load(f"{npy_dir}/labs_percentiles_{split}.npy", mmap_mode='r')
        self.labs_miss = np.load(f"{npy_dir}/labs_missingness_{split}.npy", mmap_mode='r')
        
        # Get Cardiomegaly labels (our CVD target!)
        # CheXpert encoding: 1.0=positive, 0.0=negative, -1.0=uncertain, NaN=missing
        if 'Cardiomegaly' in self.df.columns:
            labels = self.df['Cardiomegaly'].fillna(0).values
            # Binary: 1 = has cardiomegaly (positive or uncertain)
            self.labels = ((labels == 1.0) | (labels == -1.0)).astype(int)
        else:
            print(f"⚠️ Cardiomegaly column not found, using dummy labels")
            self.labels = np.zeros(len(self.df), dtype=int)
        
        print(f"\n📊 {split.upper()} split loaded:")
        print(f"   Samples: {len(self.df):,}")
        print(f"   CXR: {self.cxr.shape}")
        print(f"   ECG: {self.ecg.shape}")
        print(f"   Labs: {self.labs_pct.shape}")
        print(f"   Cardiomegaly+: {self.labels.sum():,} ({self.labels.mean()*100:.1f}%)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # CXR: (3, 320, 320) - already normalized
        cxr = torch.tensor(self.cxr[idx], dtype=torch.float32)
        
        # ECG: (1, 5000, 12) -> (12, 5000) for 1D-CNN
        ecg = torch.tensor(self.ecg[idx], dtype=torch.float32)
        ecg = ecg.squeeze(0).transpose(0, 1)  # Now (12, 5000)
        
        # Labs: percentiles (50) + missingness (50) = (100,)
        labs = np.concatenate([self.labs_pct[idx], self.labs_miss[idx]])
        labs = torch.tensor(labs, dtype=torch.float32)
        
        label = int(self.labels[idx])
        
        return cxr, ecg, labs, label


def create_dataloaders(data_dir, batch_size=32, num_workers=2):
    """Create train/val/test dataloaders with parallel loading."""
    train_ds = SymileMIMICDataset(data_dir, 'train')
    val_ds = SymileMIMICDataset(data_dir, 'val')
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"\n✅ DataLoaders created:")
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Val: {len(val_loader)} batches")
    
    return train_loader, val_loader


# ============== USAGE ==============
if __name__ == "__main__":
    # Example usage
    DATA_DIR = "/content/symile-mimic"  # Colab path
    
    train_loader, val_loader = create_dataloaders(DATA_DIR, batch_size=32)
    
    # Test one batch
    cxr, ecg, labs, labels = next(iter(train_loader))
    print(f"\n📦 Batch shapes:")
    print(f"   CXR: {cxr.shape}")   # (32, 3, 320, 320)
    print(f"   ECG: {ecg.shape}")   # (32, 12, 5000)
    print(f"   Labs: {labs.shape}") # (32, 100)
    print(f"   Labels: {labels.shape}, sum={labels.sum()}")
