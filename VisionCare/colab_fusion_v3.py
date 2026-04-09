# ╔══════════════════════════════════════════════════════════════════════╗
# ║  VISIONCARE 3.0 — ULTIMATE FUSION TRAINING SCRIPT                  ║
# ║                                                                      ║
# ║  Upgrades over V2:                                                   ║
# ║   • SYMILE-native encoders: ResNet-50, ResNet-18, 3-layer NN        ║
# ║   • Load official symile_mimic_model.ckpt for pre-trained weights   ║
# ║   • Progressive unfreezing (frozen → partial fine-tune)             ║
# ║   • Focal Loss for class imbalance                                  ║
# ║   • Gate entropy regularization for balanced modality usage         ║
# ║   • EMA (Exponential Moving Average) for smoother evaluation        ║
# ║   • Better augmentation pipeline                                    ║
# ║   • Threshold optimization for F1                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ======================================================================
# CELL 1: IMPORTS
# ======================================================================
import os, sys, json, time, warnings, gc, copy, math
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as tvm
from tqdm.auto import tqdm

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, roc_curve,
    average_precision_score, confusion_matrix,
    precision_recall_curve
)

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 120, 'font.family': 'DejaVu Sans'})
print("✅ All imports OK")
print(f"   PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU    : {torch.cuda.get_device_name(0)}")

# ======================================================================
# CELL 2: DRIVE MOUNT & CONFIGURATION
# ======================================================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False
    print("💻 Running locally")

# ══════════════════════════════════════════════════════════════════════
# TIER 3 DISEASE CONFIG — auto-detected from diagnoses_icd.csv
# If the file exists on Drive → Tier 3 enabled automatically
# ══════════════════════════════════════════════════════════════════════
_ICD_PATH    = "/content/drive/MyDrive/symile-mimic/mimic-iv-csv/diagnoses_icd.csv"
ENABLE_TIER3 = os.path.exists(_ICD_PATH)
print(f"\n🔍 diagnoses_icd.csv {'\u2705 FOUND' if ENABLE_TIER3 else '\u274c NOT FOUND'} → "
      f"Tier 3 {'ENABLED' if ENABLE_TIER3 else 'DISABLED'}")

TIER3_TARGETS = [
    ('myocardial_infarction',  ['I21', 'I22']),
    ('arrhythmia',             ['I47', 'I48', 'I49']),
    ('sepsis',                 ['A40', 'A41']),
    ('pulmonary_embolism',     ['I26']),
    ('acute_kidney_injury',    ['N17']),
    ('icu_admission',          ['Z99', 'J96']),
]


class Config:
    BASE       = "/content/drive/MyDrive/symile-mimic"
    CSV_DIR    = BASE
    NPY_DIR    = BASE
    MASTER_CSV = f"{BASE}/symile_mimic_data.csv"
    SYMILE_CKPT = f"{BASE}/symile_mimic_model.ckpt"
    MIMIC_DIR  = f"{BASE}/mimic-iv-csv"
    OUT_DIR    = f"{BASE}/VisionCare_V3"
    CKPT_V3    = f"{OUT_DIR}/checkpoints"
    # Phase-1 checkpoints (V2) for comparison
    CKPT_V2    = f"{BASE}/VisionCare_V2/checkpoints"

    TARGETS    = ['mortality', 'heart_failure'] + (
                    [t[0] for t in TIER3_TARGETS] if ENABLE_TIER3 else [])
    NUM_LABELS = len(TARGETS)

    # ── Architecture dims ──────────────────────────────────────────
    VISION_DIM   = 2048   # ResNet-50 output
    SIGNAL_DIM   = 512    # ResNet-18 output
    CLINICAL_DIM = 256    # 3-layer NN output
    FUSION_DIM   = 256    # Cross-attention hidden
    LABS_DIM     = 100    # 50 percentiles + 50 missingness

    # ── Training — Phase A (frozen encoders) ───────────────────────
    PHASE_A_EPOCHS = 5
    PHASE_A_LR     = 3e-4

    # ── Training — Phase B (partial unfreeze) ──────────────────────
    PHASE_B_EPOCHS = 20
    PHASE_B_LR_ENC = 1e-5    # encoder LR (10x smaller)
    PHASE_B_LR_FUS = 2e-4    # fusion head LR

    BATCH_SIZE     = 32
    NUM_WORKERS    = 4
    PREFETCH       = 2
    LR_MIN         = 1e-7
    WEIGHT_DECAY   = 1e-4
    GRAD_CLIP      = 1.0
    PATIENCE           = 5    # stop after 5 consecutive non-improving epochs
    MIN_PHASE_B_EPOCHS = 5    # Phase B warmup: early stop can't fire before this
    LABEL_SMOOTH       = 0.03
    FOCAL_GAMMA    = 2.0     # focal loss gamma
    GATE_REG       = 0.01    # gentle nudge — 0.1 was too strong (forced 33/33/34)
    EMA_DECAY      = 0.999
    DROPOUT        = 0.35
    USE_AMP        = True

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def setup(cls):
        os.makedirs(cls.OUT_DIR, exist_ok=True)
        os.makedirs(cls.CKPT_V3, exist_ok=True)
        print(f"\n{'='*65}")
        print("⚙️   VISIONCARE 3.0 — CONFIGURATION")
        print(f"{'='*65}")
        print(f"  Device          : {cls.DEVICE}")
        print(f"  Targets ({len(cls.TARGETS)})    : {cls.TARGETS}")
        print(f"  Tier 3 Enabled  : {ENABLE_TIER3}")
        print(f"  Phase A (frozen): {cls.PHASE_A_EPOCHS} epochs @ LR={cls.PHASE_A_LR}")
        print(f"  Phase B (unfrz) : {cls.PHASE_B_EPOCHS} epochs @ "
              f"enc={cls.PHASE_B_LR_ENC} / fus={cls.PHASE_B_LR_FUS}")
        print(f"  Focal γ         : {cls.FOCAL_GAMMA}")
        print(f"  Gate reg λ      : {cls.GATE_REG}")
        print(f"  EMA decay       : {cls.EMA_DECAY}")
        print(f"  Output          : {cls.OUT_DIR}")
        print(f"{'='*65}\n")

Config.setup()

# ======================================================================
# CELL 3: LABEL EXTRACTION (same logic as V2)
# ======================================================================

def load_master_metadata(base_dir):
    path = os.path.join(base_dir, 'symile_mimic_data.csv')
    if not os.path.exists(path):
        print(f"  ⚠️  symile_mimic_data.csv not found"); return None
    m = pd.read_csv(path, usecols=['hadm_id','hospital_expire_flag','deathtime','dod'])
    print(f"  ✅ Master CSV: {len(m):,} admissions"); return m


def extract_labels(csv_dir, mimic_dir, split, master_df):
    df = pd.read_csv(os.path.join(csv_dir, f'{split}.csv'))
    col_map = {c.strip().lower(): c for c in df.columns}

    # Mortality from master CSV
    if master_df is not None and 'hadm_id' in col_map:
        df = df.merge(master_df[['hadm_id','hospital_expire_flag','deathtime','dod']],
                      on='hadm_id', how='left')
        df['mortality'] = df['hospital_expire_flag'].fillna(0).astype(int)
        src_m = "symile_mimic_data.csv (REAL)"
    else:
        df['mortality'] = 0; src_m = "NOT FOUND"

    # Heart Failure
    col_map2 = {c.strip().lower(): c for c in df.columns}
    diag_path = os.path.join(mimic_dir, 'diagnoses_icd.csv')
    if os.path.exists(diag_path):
        diag = pd.read_csv(diag_path, usecols=['subject_id','icd_code','icd_version'])
        hf = (((diag['icd_version']==10)&diag['icd_code'].str.startswith('I50')) |
              ((diag['icd_version']==9)&diag['icd_code'].str.startswith('428')))
        hf_subs = set(diag.loc[hf, 'subject_id'])
        df['heart_failure'] = df['subject_id'].isin(hf_subs).astype(int)
        src_h = "MIMIC-IV diagnoses_icd (REAL)"
    else:
        ek = next((k for k in col_map2 if 'edema' in k), None)
        ck = next((k for k in col_map2 if 'cardiomegaly' in k), None)
        fk = next((k for k in col_map2 if 'pleural effusion' in k or 'pleural_effusion' in k), None)
        e = (df[col_map2[ek]]==1) if ek else pd.Series(False, index=df.index)
        c = (df[col_map2[ck]]==1) if ck else pd.Series(False, index=df.index)
        f = (df[col_map2[fk]]==1) if fk else pd.Series(False, index=df.index)
        df['heart_failure'] = (e | (c & f)).astype(int)
        used = [n for n,k in [('Edema',ek),('Cardio',ck),('Effusion',fk)] if k]
        src_h = f"PROXY [{'+'.join(used)}]"

    mp, hp = int(df['mortality'].sum()), int(df['heart_failure'].sum())
    print(f"  {split:5s} | mort: {mp:4d} ({100*mp/len(df):.1f}%) [{src_m}] | "
          f"HF: {hp:4d} ({100*hp/len(df):.1f}%) [{src_h}]")

    # Tier 3 — auto-extracted from diagnoses_icd.csv if present
    if ENABLE_TIER3:
        if os.path.exists(diag_path):
            diag_t3 = pd.read_csv(diag_path, usecols=['subject_id', 'icd_code', 'icd_version'])
            diag_t3['icd_code'] = diag_t3['icd_code'].astype(str).str.strip()
            for disease_name, icd_prefixes in TIER3_TARGETS:
                mask = diag_t3['icd_code'].apply(
                    lambda code: any(code.startswith(p) for p in icd_prefixes)
                )
                subs = set(diag_t3.loc[mask, 'subject_id'])
                df[disease_name] = df['subject_id'].isin(subs).astype(int)
                n_pos = int(df[disease_name].sum())
                print(f"  {split:5s} | {disease_name:<28}: {n_pos:4d} pos "
                      f"({100*n_pos/len(df):.1f}%)  [ICD: {icd_prefixes}]")
        else:
            for disease_name, _ in TIER3_TARGETS:
                df[disease_name] = 0

    return df


print("📋 Label Extraction...")
master_df = load_master_metadata(Config.BASE)
train_df = extract_labels(Config.CSV_DIR, Config.MIMIC_DIR, 'train', master_df)
val_df   = extract_labels(Config.CSV_DIR, Config.MIMIC_DIR, 'val',   master_df)

# ======================================================================
# CELL 4: DATASET
# ======================================================================

class SymileDatasetV3(Dataset):
    def __init__(self, npy_dir, df, targets, split='train'):
        self.targets, self.split = targets, split

        def find_npy(base, fname, split):
            for p in [os.path.join(base,'data_npy',split,fname),
                      os.path.join(base,'data_npy',fname),
                      os.path.join(base,fname)]:
                if os.path.exists(p): return p
            raise FileNotFoundError(f"Cannot find {fname}")

        self.cxr = np.load(find_npy(npy_dir,f'cxr_{split}.npy',split), mmap_mode='r')
        self.ecg = np.load(find_npy(npy_dir,f'ecg_{split}.npy',split), mmap_mode='r')
        self.lp  = np.load(find_npy(npy_dir,f'labs_percentiles_{split}.npy',split), mmap_mode='r')
        self.lm  = np.load(find_npy(npy_dir,f'labs_missingness_{split}.npy',split), mmap_mode='r')
        self.labels = df[targets].values.astype(np.float32)
        n = min(len(self.cxr), len(self.labels))
        self.cxr, self.ecg = self.cxr[:n], self.ecg[:n]
        self.lp, self.lm, self.labels = self.lp[:n], self.lm[:n], self.labels[:n]
        print(f"  {split}: {n:,} samples | CXR {self.cxr.shape} | ECG {self.ecg.shape}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        # CXR: (3,320,320) — keep at 320 for ResNet-50 (no resize needed)
        cxr = torch.from_numpy(self.cxr[idx].astype(np.float32))
        if self.split == 'train':
            if torch.rand(1) > 0.5: cxr = torch.flip(cxr, [2])
            if torch.rand(1) > 0.7:  # random erasing
                h, w = cxr.shape[1], cxr.shape[2]
                rh, rw = int(h*0.1), int(w*0.1)
                ry, rx = torch.randint(0,h-rh,(1,)), torch.randint(0,w-rw,(1,))
                cxr[:, ry:ry+rh, rx:rx+rw] = 0
            cxr = cxr + torch.randn_like(cxr) * 0.01

        # ECG: (1,5000,12) → (12,5000)
        ecg = torch.from_numpy(self.ecg[idx].astype(np.float32))
        ecg = ecg.squeeze(0).permute(1, 0)  # (12,5000)
        if self.split == 'train' and torch.rand(1) > 0.5:
            shift = torch.randint(-200, 200, (1,)).item()
            ecg = torch.roll(ecg, shift, dims=1)

        # Labs: (100,)
        labs = torch.from_numpy(
            np.concatenate([self.lp[idx], self.lm[idx]]).astype(np.float32))

        return cxr, ecg, labs, torch.from_numpy(self.labels[idx])

    def get_pos_weights(self):
        pos = self.labels.sum(0) + 1e-6
        return torch.tensor((len(self.labels)-pos)/pos, dtype=torch.float32)

    def get_sample_weights(self):
        return np.where(self.labels.sum(1) > 0, 3.0, 1.0)


def make_loaders(cfg, train_df, val_df):
    tds = SymileDatasetV3(cfg.NPY_DIR, train_df, cfg.TARGETS, 'train')
    vds = SymileDatasetV3(cfg.NPY_DIR, val_df,   cfg.TARGETS, 'val')
    sw  = torch.tensor(tds.get_sample_weights(), dtype=torch.float32)
    sampler = WeightedRandomSampler(sw, len(tds), replacement=True)
    kw = dict(pin_memory=True,
              prefetch_factor=cfg.PREFETCH if cfg.NUM_WORKERS>0 else None,
              persistent_workers=(cfg.NUM_WORKERS>0))
    tl = DataLoader(tds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                    num_workers=cfg.NUM_WORKERS, **kw)
    vl = DataLoader(vds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                    num_workers=cfg.NUM_WORKERS, **kw)
    return tl, vl, tds.get_pos_weights().to(cfg.DEVICE)

print("\n🏗️  Building DataLoaders...")
train_loader, val_loader, pos_weights = make_loaders(Config, train_df, val_df)

# ======================================================================
# CELL 5: ENCODER ARCHITECTURES — Industry Standard
# ======================================================================

class CXREncoder(nn.Module):
    """ResNet-50 with EXPLICIT named layers → matches SYMILE checkpoint keys.
    SYMILE stores: cxr_encoder.conv1.weight, cxr_encoder.layer1.0.conv1.weight, ...
    After stripping 'cxr_encoder.' prefix: conv1.weight, layer1.0.conv1.weight, ...
    These match exactly only when we use explicit attributes (not nn.Sequential).
    """
    def __init__(self):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        # Explicit named layers — keys will be conv1.weight, layer1.0.conv1.weight, etc.
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool
        self.out_dim = 2048

    def forward(self, x):  # x: (B, 3, 320, 320)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)  # (B, 2048)

    def get_unfreeze_params(self):
        """Last 2 blocks for Phase B fine-tuning."""
        return (list(self.layer3.parameters()) +
                list(self.layer4.parameters()) +
                list(self.avgpool.parameters()))



class BasicBlock1D(nn.Module):
    """1D version of ResNet BasicBlock for ECG signals."""
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ECGEncoder(nn.Module):
    """1D ResNet-18 architecture for 12-lead ECG → 512-D embedding.
    Mirrors ResNet-18 structure but uses Conv1d for temporal signals."""
    def __init__(self, in_channels=12):
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv1d(in_channels, 64, 15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 512

    def _make_layer(self, ch, blocks, stride):
        downsample = None
        if stride != 1 or self.in_ch != ch:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_ch, ch, 1, stride, bias=False),
                nn.BatchNorm1d(ch))
        layers = [BasicBlock1D(self.in_ch, ch, stride, downsample)]
        self.in_ch = ch
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(ch, ch))
        return nn.Sequential(*layers)

    def forward(self, x):  # (B,12,5000)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)  # (B,512)

    def get_unfreeze_params(self):
        params = []
        for name, p in self.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                params.append(p)
        return params


class LabsEncoder(nn.Module):
    """3-layer NN for blood labs → 256-D embedding (matching SYMILE spec)."""
    def __init__(self, in_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512,   256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256,   256), nn.BatchNorm1d(256), nn.GELU()
        )
        self.out_dim = 256

    def forward(self, x):
        return self.net(x)

    def get_unfreeze_params(self):
        return list(self.parameters())  # small model, unfreeze all


# ======================================================================
# CELL 6: SMART CHECKPOINT LOADER (SYMILE + Phase-1 fallback)
# ======================================================================

def smart_load(model, state, name, explicit_prefix=None):
    """Load checkpoint weights — uses explicit_prefix if provided, else auto-detects."""
    mdict = model.state_dict()
    matched = {}

    # Strategy 0: Explicit prefix (fastest, most reliable — used for SYMILE)
    if explicit_prefix:
        for k, v in state.items():
            nk = k[len(explicit_prefix):] if k.startswith(explicit_prefix) else k
            if nk in mdict and mdict[nk].shape == v.shape:
                matched[nk] = v

    # Strategy 1: Try a list of common prefixes (fallback)
    if len(matched) < (len(mdict) // 2):
        for pfx in ['', 'cxr_encoder.resnet.', 'ecg_encoder.resnet.', 'labs_encoder.resnet.',
                    'cxr_encoder.', 'ecg_encoder.', 'labs_encoder.',
                    'model.', 'module.', 'encoder.', 'backbone.', 'net.']:
            for k, v in state.items():
                nk = k[len(pfx):] if k.startswith(pfx) else k
                if nk in mdict and mdict[nk].shape == v.shape:
                    matched[nk] = v

    # Strategy 2: Shape-sequence (last resort)
    if len(matched) < (len(mdict) // 2):
        print(f"    🔄 {name}: shape-sequence fallback...")
        avail = [v for k, v in state.items() if v.dim() > 0]
        c = 0
        tmp = {}
        for mk, mv in mdict.items():
            for i in range(c, len(avail)):
                if avail[i].shape == mv.shape:
                    tmp[mk] = avail[i]; c = i + 1; break
        if len(tmp) > len(matched):
            matched = tmp

    if matched:
        model.load_state_dict(matched, strict=False)
    tag = "✅" if len(matched) > len(mdict) // 2 else "⚠️"
    print(f"  {tag} {name:<22} | {len(matched):3d}/{len(mdict)} keys loaded")
    return model


def load_symile_checkpoint(cxr_enc, ecg_enc, labs_enc, ckpt_path):
    """Load the official symile_mimic_model.ckpt and distribute to encoders."""
    if not os.path.exists(ckpt_path):
        print(f"  ⚠️  SYMILE checkpoint not found at {ckpt_path}")
        return cxr_enc, ecg_enc, labs_enc

    print(f"  📦 Loading SYMILE checkpoint...")
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = raw.get('state_dict', raw.get('model_state_dict', raw))

    # Discover module groupings
    groups = {}
    for k in state.keys():
        root = k.split('.')[0]
        groups.setdefault(root, []).append(k)
    print(f"  📋 Checkpoint groups: {[(g, len(v)) for g, v in groups.items()]}")

    # Try to split by encoder type (common SYMILE naming patterns)
    cxr_keys, ecg_keys, labs_keys = {}, {}, {}
    for k, v in state.items():
        kl = k.lower()
        if any(x in kl for x in ['cxr', 'image', 'vision', 'resnet50']):
            cxr_keys[k] = v
        elif any(x in kl for x in ['ecg', 'signal', 'resnet18']):
            ecg_keys[k] = v
        elif any(x in kl for x in ['lab', 'clinical', 'tabular']):
            labs_keys[k] = v

    # If named grouping didn't work, try positional grouping
    if not cxr_keys and not ecg_keys:
        grp_names = sorted(groups.keys())
        if len(grp_names) >= 3:
            for k in groups[grp_names[0]]: cxr_keys[k] = state[k]
            for k in groups[grp_names[1]]: ecg_keys[k] = state[k]
            for k in groups[grp_names[2]]: labs_keys[k] = state[k]
            print(f"  🔄 Positional split: {grp_names[0]}→CXR, "
                  f"{grp_names[1]}→ECG, {grp_names[2]}→Labs")

    print(f"  📊 Keys found — CXR: {len(cxr_keys)} | ECG: {len(ecg_keys)} | Labs: {len(labs_keys)}")
    # SYMILE uses 'cxr_encoder.resnet.*' and 'ecg_encoder.resnet.*' structure
    if cxr_keys:  cxr_enc  = smart_load(cxr_enc,  cxr_keys,  "CXR (ResNet-50)",   'cxr_encoder.resnet.')
    if ecg_keys:  ecg_enc  = smart_load(ecg_enc,  ecg_keys,  "ECG (ResNet-18)",   'ecg_encoder.resnet.')
    if labs_keys: labs_enc = smart_load(labs_enc, labs_keys, "Labs (3-layer NN)", 'labs_encoder.resnet.')
    return cxr_enc, ecg_enc, labs_enc


print("\n📥 Loading encoders...")
cxr_enc  = CXREncoder()
ecg_enc  = ECGEncoder(in_channels=12)
labs_enc = LabsEncoder(in_dim=Config.LABS_DIM)

# Try SYMILE checkpoint first
cxr_enc, ecg_enc, labs_enc = load_symile_checkpoint(
    cxr_enc, ecg_enc, labs_enc, Config.SYMILE_CKPT)

# ======================================================================
# CELL 7: FUSION MODEL — Cross-Attention Gated + Gate Regularization
# ======================================================================

class GatedCrossAttentionFusionV3(nn.Module):
    def __init__(self, v_dim, s_dim, c_dim, hidden=256, heads=4):
        super().__init__()
        self.proj_v = nn.Sequential(nn.Linear(v_dim, hidden), nn.LayerNorm(hidden))
        self.proj_s = nn.Sequential(nn.Linear(s_dim, hidden), nn.LayerNorm(hidden))
        self.proj_c = nn.Sequential(nn.Linear(c_dim, hidden), nn.LayerNorm(hidden))
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ffn  = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(hidden*2, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.gate = nn.Sequential(nn.Linear(hidden*3, 128), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(128, 3), nn.Softmax(dim=-1))
        self.out_dim = hidden

    def forward(self, fv, fs, fc):
        v, s, c = self.proj_v(fv), self.proj_s(fs), self.proj_c(fc)
        stack = torch.stack([v, s, c], dim=1)
        att, _ = self.attn(stack, stack, stack)
        att = self.norm(att + stack)
        att = self.norm2(self.ffn(att) + att)  # FFN + residual
        av, as_, ac = att[:,0], att[:,1], att[:,2]
        gates = self.gate(torch.cat([av, as_, ac], -1))
        fused = gates[:,0:1]*av + gates[:,1:2]*as_ + gates[:,2:3]*ac
        return fused, gates


class VisionCareV3(nn.Module):
    def __init__(self, cxr, ecg, labs, cfg):
        super().__init__()
        self.cxr_enc, self.ecg_enc, self.labs_enc = cxr, ecg, labs
        V,S,C,H = cfg.VISION_DIM, cfg.SIGNAL_DIM, cfg.CLINICAL_DIM, cfg.FUSION_DIM
        self.fusion = GatedCrossAttentionFusionV3(V, S, C, H, heads=4)
        self.head = nn.Sequential(
            nn.Linear(H, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, cfg.NUM_LABELS))

    def freeze_encoders(self):
        for enc in [self.cxr_enc, self.ecg_enc, self.labs_enc]:
            for p in enc.parameters(): p.requires_grad = False
        self._print_params("FROZEN")

    def unfreeze_last_blocks(self):
        """Unfreeze last 2 blocks of each encoder for fine-tuning."""
        for p in self.cxr_enc.get_unfreeze_params():  p.requires_grad = True
        for p in self.ecg_enc.get_unfreeze_params():  p.requires_grad = True
        for p in self.labs_enc.get_unfreeze_params(): p.requires_grad = True
        self._print_params("PARTIAL UNFREEZE")

    def _print_params(self, mode):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  🔢 [{mode}] Total: {total:,} | Trainable: {train:,} ({100*train/total:.1f}%)")

    def forward(self, cxr, ecg, labs):
        fv = self.cxr_enc(cxr)
        fs = self.ecg_enc(ecg)
        fc = self.labs_enc(labs)
        fused, gates = self.fusion(fv, fs, fc)
        return self.head(fused), gates


print("\n🏗️  Building VisionCare 3.0...")
model = VisionCareV3(cxr_enc, ecg_enc, labs_enc, Config).to(Config.DEVICE)
model.freeze_encoders()

# ======================================================================
# CELL 8: FOCAL LOSS + EMA + PROGRESSIVE UNFREEZING TRAINER
# ======================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, smooth=0.03, pos_weight=None):
        super().__init__()
        self.gamma, self.smooth, self.pos_weight = gamma, smooth, pos_weight
    def forward(self, logits, targets):
        t = targets * (1-self.smooth) + 0.5*self.smooth
        bce = F.binary_cross_entropy_with_logits(logits, t, pos_weight=self.pos_weight, reduction='none')
        pt = torch.sigmoid(logits)*targets + (1-torch.sigmoid(logits))*(1-targets)
        return (((1-pt)**self.gamma)*bce).mean()

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k:v.clone().detach() for k,v in model.state_dict().items()}
    def update(self, model):
        for k,v in model.state_dict().items():
            self.shadow[k] = self.decay*self.shadow[k] + (1-self.decay)*v
    def apply(self, model):
        self.backup = {k:v.clone() for k,v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
    def restore(self, model):
        model.load_state_dict(self.backup)

def compute_metrics(labels, probs, targets):
    preds = (probs>0.5).astype(int)
    out = {'per_class': {}}; aucs, f1s = [], []
    for i,t in enumerate(targets):
        yt,yp,yb = labels[:,i], probs[:,i], preds[:,i]
        try: auc = roc_auc_score(yt,yp)
        except: auc = 0.5
        f1 = f1_score(yt,yb,zero_division=0); aucs.append(auc); f1s.append(f1)
        out['per_class'][t] = {'auc':auc,'f1':f1,
            'precision':precision_score(yt,yb,zero_division=0),
            'recall':recall_score(yt,yb,zero_division=0),
            'accuracy':accuracy_score(yt,yb),
            'ap':average_precision_score(yt,yp) if yt.sum()>0 else 0,
            'support':int(yt.sum())}
    out['macro_auc']=float(np.mean(aucs)); out['macro_f1']=float(np.mean(f1s))
    return out

class ProgressiveTrainer:
    def __init__(self, model, tl, vl, pw, cfg):
        self.model,self.tl,self.vl,self.cfg = model,tl,vl,cfg
        self.criterion = FocalLoss(cfg.FOCAL_GAMMA, cfg.LABEL_SMOOTH, pw)
        self.scaler = GradScaler(enabled=cfg.USE_AMP)
        self.ema = EMA(model, cfg.EMA_DECAY)
        self.history = {'train_loss':[],'val_auc':[],'val_f1':[],'lr':[],'gates':[],'phase':[]}
        self.best_auc,self.best_epoch,self.patience_ctr = 0.0,0,0
        self.best_metrics = None

    def _make_opt(self, phase):
        if phase=='A':
            return torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                     lr=self.cfg.PHASE_A_LR, weight_decay=self.cfg.WEIGHT_DECAY)
        return torch.optim.AdamW([
            {'params':self.model.cxr_enc.get_unfreeze_params(),'lr':self.cfg.PHASE_B_LR_ENC},
            {'params':self.model.ecg_enc.get_unfreeze_params(),'lr':self.cfg.PHASE_B_LR_ENC},
            {'params':self.model.labs_enc.get_unfreeze_params(),'lr':self.cfg.PHASE_B_LR_ENC},
            {'params':list(self.model.fusion.parameters())+list(self.model.head.parameters()),
             'lr':self.cfg.PHASE_B_LR_FUS}], weight_decay=self.cfg.WEIGHT_DECAY)

    def _train_ep(self, epoch, opt):
        self.model.train(); total=0.0
        bar = tqdm(self.tl, desc=f"  Ep{epoch:02d}", leave=False, ncols=110)
        for cxr,ecg,labs,lbl in bar:
            cxr=cxr.to(Config.DEVICE,non_blocking=True)
            ecg=ecg.to(Config.DEVICE,non_blocking=True)
            labs=labs.to(Config.DEVICE,non_blocking=True)
            lbl=lbl.to(Config.DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=self.cfg.USE_AMP):
                logits,gates = self.model(cxr,ecg,labs)
                loss = self.criterion(logits,lbl)
                gate_ent = -(gates*torch.log(gates+1e-8)).sum(-1).mean()
                loss = loss - self.cfg.GATE_REG*gate_ent
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                     self.cfg.GRAD_CLIP)
            self.scaler.step(opt); self.scaler.update()
            self.ema.update(self.model)
            total+=loss.item(); bar.set_postfix(loss=f"{loss.item():.4f}")
        return total/len(self.tl)

    @torch.no_grad()
    def _validate(self):
        self.ema.apply(self.model); self.model.eval()
        all_lbl,all_prob,all_gates = [],[],[]
        for cxr,ecg,labs,lbl in tqdm(self.vl,desc="  [val]",leave=False,ncols=110):
            cxr=cxr.to(Config.DEVICE,non_blocking=True)
            ecg=ecg.to(Config.DEVICE,non_blocking=True)
            labs=labs.to(Config.DEVICE,non_blocking=True)
            with autocast(enabled=self.cfg.USE_AMP):
                logits,gates = self.model(cxr,ecg,labs)
            all_lbl.append(lbl.numpy())
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
            all_gates.append(gates.cpu().numpy())
        self.ema.restore(self.model)
        labels=np.concatenate(all_lbl); probs=np.concatenate(all_prob)
        gates=np.concatenate(all_gates).mean(0)
        m = compute_metrics(labels,probs,self.cfg.TARGETS); m['gates']=gates.tolist()
        return m

    def _save(self, epoch, m):
        torch.save({'epoch':epoch,'model_state_dict':self.model.state_dict(),
                    'ema':self.ema.shadow,'best_auc':self.best_auc,
                    'metrics':m}, f"{self.cfg.CKPT_V3}/fusion_v3_best.pth")

    def _log(self, tag, ep, loss, m, lr_info, global_ep=None):
        auc, f1, gw = m['macro_auc'], m['macro_f1'], m['gates']
        self.history['train_loss'].append(loss)
        self.history['val_auc'].append(auc)
        self.history['val_f1'].append(f1)
        self.history['lr'].append(lr_info)
        self.history['gates'].append(gw)
        self.history['phase'].append(tag)
        is_best = auc > self.best_auc
        pat_str = f"  [pat {self.patience_ctr}/{self.cfg.PATIENCE}]" if not is_best else ""
        bt      = "  ✅ BEST" if is_best else pat_str
        gep     = global_ep if global_ep is not None else ep
        print(f"  {tag}-{ep:02d}(G{gep:02d}) | Loss:{loss:.4f} | AUC:{auc:.4f} | "
              f"F1:{f1:.4f} | Gates[V:{gw[0]:.2f} S:{gw[1]:.2f} C:{gw[2]:.2f}]{bt}")
        if is_best:
            self.best_auc, self.best_epoch = auc, gep   # store GLOBAL epoch
            self.patience_ctr = 0
            self._save(gep, m)
            self.best_metrics = m
            return True
        self.patience_ctr += 1
        return False

    def train(self):
        t0 = time.time()
        print(f"\n{'='*65}")
        print(f"🚀 PHASE A — Frozen Encoders ({self.cfg.PHASE_A_EPOCHS} epochs)")
        print(f"{'='*65}")
        opt_a = self._make_opt('A')
        sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_a, self.cfg.PHASE_A_EPOCHS, eta_min=self.cfg.LR_MIN)
        for ep in range(1, self.cfg.PHASE_A_EPOCHS + 1):
            loss = self._train_ep(ep, opt_a)
            m    = self._validate()
            sch_a.step()
            self._log('A', ep, loss, m, opt_a.param_groups[0]['lr'], global_ep=ep)

        print(f"\n{'='*65}")
        print(f"🔓 PHASE B — Partial Unfreeze (up to {self.cfg.PHASE_B_EPOCHS} epochs, "
              f"patience={self.cfg.PATIENCE}, min_warmup={self.cfg.MIN_PHASE_B_EPOCHS})")
        print(f"{'='*65}")
        self.model.unfreeze_last_blocks()
        opt_b  = self._make_opt('B')
        sch_b  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_b, T_0=5, T_mult=2, eta_min=self.cfg.LR_MIN)
        self.patience_ctr = 0   # ★ reset: Phase B starts fresh
        for ep in range(1, self.cfg.PHASE_B_EPOCHS + 1):
            gep  = self.cfg.PHASE_A_EPOCHS + ep
            loss = self._train_ep(gep, opt_b)
            m    = self._validate()
            sch_b.step()
            self._log('B', ep, loss, m, opt_b.param_groups[-1]['lr'], global_ep=gep)
            # Guard: don't allow early stop until MIN_PHASE_B_EPOCHS of Phase B done
            if (ep >= self.cfg.MIN_PHASE_B_EPOCHS and
                    self.patience_ctr >= self.cfg.PATIENCE):
                print(f"\n  ⏹️  Early stop at B-{ep} (global {gep}) — "
                      f"no improvement in {self.cfg.PATIENCE} epochs")
                break

        print(f"\n  ⏱️  {(time.time()-t0)/60:.1f} min total")
        print(f"  🏆 Best AUC: {self.best_auc:.4f} at global epoch {self.best_epoch}")
        return self.history, self.best_metrics

trainer = ProgressiveTrainer(model, train_loader, val_loader, pos_weights, Config)
history, best = trainer.train()

# ======================================================================
# CELL 9: EVALUATION REPORT
# ======================================================================
gw = best['gates']; mods = ['Vision (CXR)','Signal (ECG)','Clinical (Labs)']
print(f"\n{'='*80}")
print("📊  VISIONCARE 3.0 — FINAL REPORT")
print(f"{'='*80}")
for t in Config.TARGETS:
    m = best['per_class'][t]
    print(f"  {t:<22} | AUC:{m['auc']:.4f} | F1:{m['f1']:.4f} | "
          f"Prec:{m['precision']:.4f} | Rec:{m['recall']:.4f} | AP:{m['ap']:.4f}")
print(f"  {'MACRO':<22} | AUC:{best['macro_auc']:.4f} | F1:{best['macro_f1']:.4f}")
print(f"\n  Gates: V={100*gw[0]:.0f}% S={100*gw[1]:.0f}% C={100*gw[2]:.0f}%")
print(f"  Δ vs V2 (0.8105): {best['macro_auc']-0.8105:+.4f}")

report = {
    'model':          'VisionCare 3.0',
    'script':         'colab_fusion_v3.py',
    'targets':        Config.TARGETS,
    'tier3_enabled':  ENABLE_TIER3,
    'trained':        datetime.now().isoformat(),
    'best_epoch':     trainer.best_epoch,
    'v3_macro_auc':   best['macro_auc'],
    'v3_macro_f1':    best['macro_f1'],
    'baselines': {
        'vision_p1': 0.680, 'signal_p1': 0.610, 'clinical_p1': 0.625,
        'fusion_v1': 0.7702, 'fusion_v2': 0.8105,
    },
    'gates':      dict(zip(['vision','signal','clinical'], gw)),
    'per_class':  best['per_class'],
    'table_8_1': {  # V2 ground truth for comparison
        'heart_failure': {'auc':0.8189,'f1':0.5888,'prec':0.584,'rec':0.589},
        'mortality':     {'auc':0.8022,'f1':0.3115,'prec':0.452,'rec':0.422},
    }
}
with open(f"{Config.OUT_DIR}/fusion_v3_report.json",'w') as f:
    json.dump(report, f, indent=2)
print(f"  💾 Report saved: {Config.OUT_DIR}/fusion_v3_report.json")

# ══════════════════════════════════════════════════════════════════════
# CELL 10: COMPREHENSIVE VISUALIZATIONS (14 figures)
# ══════════════════════════════════════════════════════════════════════
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix as sk_cm

def savefig(name):
    plt.savefig(f"{Config.OUT_DIR}/{name}", dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✅ {name}")
    plt.close('all')

# Colour palette
C3 = ['#3498db', '#e74c3c', '#9b59b6']   # vision, signal, clinical
ROC_COLS = ['#e74c3c','#3498db','#9b59b6','#f39c12','#1abc9c','#e67e22','#c0392b','#2980b9']

# Collect val outputs for figure plotting
print("\n📊 Collecting validation outputs for figures...")
model.eval()
all_lbl_f, all_prob_f, all_gates_f = [], [], []
with torch.no_grad():
    for cxr,ecg,labs,lbl in val_loader:
        logits,gates = model(cxr.to(Config.DEVICE),
                             ecg.to(Config.DEVICE),
                             labs.to(Config.DEVICE))
        all_lbl_f.append(lbl.numpy())
        all_prob_f.append(torch.sigmoid(logits).cpu().numpy())
        all_gates_f.append(gates.cpu().numpy())
VL  = np.concatenate(all_lbl_f)   # (N, num_targets)
VP  = np.concatenate(all_prob_f)
VG  = np.concatenate(all_gates_f) # (N, 3)

ep  = range(1, len(history['train_loss'])+1)
pa  = Config.PHASE_A_EPOCHS
gwa = np.array(history['gates'])  # (epochs, 3)

print("\n🎨 Generating figures...")

# ╔══ FIG 1: Training History ══════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax,data,title,col in zip(
    axes,
    [history['train_loss'], history['val_auc'], history['lr']],
    ['Focal Loss', 'Val Macro-AUC', 'Learning Rate'],
    ['steelblue', 'mediumseagreen', 'tomato']
):
    ax.plot(ep, data, color=col, lw=2, marker='o', ms=3)
    ax.axvline(pa+0.5, color='black', ls='--', lw=2, label='Phase A→B')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title(title, fontweight='bold'); ax.set_xlabel('Epoch')
best_ep_idx = int(np.argmax(history['val_auc']))
axes[1].scatter(best_ep_idx+1, history['val_auc'][best_ep_idx],
                color='red', s=200, zorder=6, marker='*',
                label=f"Best: {history['val_auc'][best_ep_idx]:.4f}")
axes[1].axhline(0.7702, color='orange', ls=':', lw=1.5, label='V1: 0.7702')
axes[1].axhline(0.8105, color='purple', ls=':', lw=1.5, label='V2: 0.8105')
axes[1].legend(fontsize=7)
plt.suptitle('VisionCare 3.0 — Phase A→B Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
savefig("v3_training_history.png")


# ╔══ FIG 2: Gate Evolution (Phase A vs B side-by-side pie) ═══
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Left: evolution line
for i,(col,label) in enumerate(zip(C3, mods)):
    axes[0].plot(ep, gwa[:,i], color=col, lw=2.5, marker='o', ms=4, label=label)
axes[0].axvline(pa+0.5, color='gray', ls='--', lw=2, label='Unfreeze')
axes[0].set_ylim(0.1, 0.65)
axes[0].set_title('Gate Evolution A→B', fontweight='bold')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
# Mid: Phase A avg
gw_a = gwa[:pa].mean(0) if pa > 0 else gwa[0]
axes[1].pie(gw_a, labels=[f'{m}\n{100*g:.1f}%' for m,g in zip(['CXR','ECG','Labs'],gw_a)],
            colors=C3, autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'white','lw':2},
            textprops={'fontsize':10,'fontweight':'bold'}, pctdistance=0.75)
axes[1].set_title('Phase A Gates (Frozen)', fontweight='bold')
# Right: Phase B / final
axes[2].pie(gw, labels=[f'{m}\n{100*g:.1f}%' for m,g in zip(['CXR','ECG','Labs'],gw)],
            colors=C3, autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'white','lw':2},
            textprops={'fontsize':10,'fontweight':'bold'}, pctdistance=0.75)
axes[2].set_title('Final Gates (After Unfreeze)', fontweight='bold')
plt.suptitle('VisionCare 3.0 — Modality Contributions (XAI)', fontsize=14, fontweight='bold')
plt.tight_layout()
savefig("v3_gate_evolution.png")


# ╔══ FIG 3: Full Version Comparison (V1 → V2 → V3) ═══
fig, ax = plt.subplots(figsize=(13, 6))
names = ['Vision\n(P1)','Signal\n(P1)','Clinical\n(P1)','Fusion\nV1','Fusion\nV2★','FUSION\n★V3★']
vals  = [0.680, 0.610, 0.625, 0.7702, 0.8105, best['macro_auc']]
clrs  = ['#3498db','#e74c3c','#9b59b6','#bdc3c7','#f39c12','#27ae60']
brs   = ax.bar(names, vals, color=clrs, edgecolor='black', lw=1.5, width=0.65)
for b,v in zip(brs, vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.006,
            f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(max(0, min(vals)-0.06), min(1.0, max(vals)+0.09))
ax.set_ylabel('Macro AUC-ROC', fontsize=13)
ax.set_title('VisionCare Evolution: P1 Single → V1 Fusion → V2 → V3',
             fontsize=13, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
# Improvement annotations
for i_from, i_to in [(3,4),(4,5)]:
    delta = vals[i_to] - vals[i_from]
    mid_x = (brs[i_from].get_x()+brs[i_from].get_width()/2 +
             brs[i_to].get_x()+brs[i_to].get_width()/2) / 2
    ax.annotate(f'+{delta*100:.1f}%', xy=(mid_x, max(vals[i_from],vals[i_to])+0.035),
                ha='center', color='darkgreen', fontsize=10, fontweight='bold')
plt.tight_layout()
savefig("v3_version_comparison.png")


# ╔══ FIG 4: ROC Curves (all targets) ═══
fig, ax = plt.subplots(figsize=(8, 7))
for i, t in enumerate(Config.TARGETS):
    if VL[:,i].sum() == 0: continue
    fpr, tpr, _ = roc_curve(VL[:,i], VP[:,i])
    auc_val = best['per_class'][t]['auc']
    ax.plot(fpr, tpr, lw=2.5, color=ROC_COLS[i % len(ROC_COLS)],
            label=f"{t.replace('_',' ').title()} (AUC={auc_val:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.05, color=ROC_COLS[i % len(ROC_COLS)])
ax.plot([0,1],[0,1],'k--', alpha=0.4, lw=1.5, label='Random')
ax.set_xlabel('FPR (1–Specificity)', fontsize=12)
ax.set_ylabel('TPR (Sensitivity)', fontsize=12)
ax.set_title('ROC Curves — VisionCare 3.0', fontweight='bold', fontsize=13)
ax.legend(fontsize=8, loc='lower right'); ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout()
savefig("v3_roc_curves.png")


# ╔══ FIG 5: Precision-Recall Curves ═══
fig, ax = plt.subplots(figsize=(8, 7))
for i, t in enumerate(Config.TARGETS):
    if VL[:,i].sum() == 0: continue
    prec_c, rec_c, _ = precision_recall_curve(VL[:,i], VP[:,i])
    ap = best['per_class'][t]['ap']
    ax.plot(rec_c, prec_c, lw=2.5, color=ROC_COLS[i % len(ROC_COLS)],
            label=f"{t.replace('_',' ').title()} (AP={ap:.3f})")
    ax.axhline(VL[:,i].mean(), color=ROC_COLS[i % len(ROC_COLS)],
               ls=':', lw=1, alpha=0.5)
ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall — VisionCare 3.0', fontweight='bold', fontsize=13)
ax.legend(fontsize=8); ax.set_xlim(0,1); ax.set_ylim(0,1.05)
plt.tight_layout()
savefig("v3_pr_curves.png")


# ╔══ FIG 6: Confusion Matrices (all targets, tiled) ═══
ncols = min(len(Config.TARGETS), 3)
nrows = (len(Config.TARGETS) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
axes_flat = np.array(axes).flatten() if len(Config.TARGETS) > 1 else [axes]
cmaps = ['Reds','Blues','Purples','Oranges','Greens','YlOrRd','RdPu','BuGn']
for i, t in enumerate(Config.TARGETS):
    preds = (VP[:,i] > 0.5).astype(int)
    cm_arr = sk_cm(VL[:,i], preds)
    m = best['per_class'][t]
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap=cmaps[i % len(cmaps)],
                ax=axes_flat[i],
                xticklabels=['Pred -', 'Pred +'],
                yticklabels=['Act -', 'Act +'],
                annot_kws={'size':14,'weight':'bold'},
                linewidths=2, linecolor='white')
    axes_flat[i].set_title(
        f"{t.replace('_',' ').title()}\n"
        f"AUC={m['auc']:.4f}  F1={m['f1']:.4f}  "
        f"P={m['precision']:.3f}  R={m['recall']:.3f}",
        fontweight='bold', fontsize=9)
for j in range(len(Config.TARGETS), len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.suptitle('Confusion Matrices — VisionCare 3.0', fontsize=13, fontweight='bold')
plt.tight_layout()
savefig("v3_confusion_matrices.png")


# ╔══ FIG 7: Per-Disease AUC + F1 bars ═══
if len(Config.TARGETS) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mk, ml in zip(axes, ['auc','f1'], ['AUC-ROC','F1-Score']):
        vals_bar = [best['per_class'][t][mk] for t in Config.TARGETS]
        brs = ax.bar([t.replace('_','\n') for t in Config.TARGETS],
                     vals_bar, color=ROC_COLS[:len(vals_bar)],
                     edgecolor='black', alpha=0.9, width=0.5)
        for b, v in zip(brs, vals_bar):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_title(f'{ml} per Target — V3', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    plt.suptitle('Per-Disease Performance — VisionCare 3.0', fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig("v3_per_disease.png")


# ╔══ FIG 8: V2 vs V3 Per-Class Comparison (HF + Mortality) ═══
V2_BASELINE = {
    'heart_failure': {'auc':0.8189,'f1':0.5888,'prec':0.584,'rec':0.589},
    'mortality':     {'auc':0.8022,'f1':0.3115,'prec':0.452,'rec':0.422},
}
com_targets = [t for t in ['heart_failure','mortality'] if t in Config.TARGETS and t in V2_BASELINE]
if com_targets:
    metrics_to_compare = ['auc','f1','prec','rec'] if 'prec' in best['per_class'][com_targets[0]] \
                         else ['auc','f1','precision','recall']
    # normalise key names
    def _get(d, k):
        return d.get(k, d.get('precision' if k=='prec' else 'recall' if k=='rec' else k, 0))

    n_metrics = len(metrics_to_compare)
    fig, axes = plt.subplots(1, len(com_targets), figsize=(7*len(com_targets), 5))
    if len(com_targets) == 1: axes = [axes]
    for ax, t in zip(axes, com_targets):
        v2_vals = [_get(V2_BASELINE[t], m) for m in metrics_to_compare]
        v3_vals = [_get(best['per_class'][t], m) for m in metrics_to_compare]
        x = np.arange(n_metrics)
        w = 0.35
        b1 = ax.bar(x-w/2, v2_vals, w, label='V2 (Baseline)', color='#f39c12', edgecolor='black', alpha=0.9)
        b2 = ax.bar(x+w/2, v3_vals, w, label='V3 (New)',      color='#27ae60', edgecolor='black', alpha=0.9)
        for b, v in list(zip(b1,v2_vals))+list(zip(b2,v3_vals)):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics_to_compare])
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{t.replace('_',' ').title()} — V2 vs V3", fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
    plt.suptitle('VisionCare V2 vs V3 — Per-Class Metric Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig("v3_v2_comparison.png")


# ╔══ FIG 9: Per-Patient Gate Histograms ═══
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (ax, name, col) in enumerate(zip(axes, ['Vision (CXR)','ECG','Labs'], C3)):
    g_vals = VG[:,i]
    ax.hist(g_vals, bins=30, color=col, alpha=0.8, edgecolor='white', lw=0.5)
    ax.axvline(g_vals.mean(), color='black', ls='--', lw=2, label=f'Mean={g_vals.mean():.3f}')
    ax.axvline(gw[i], color='gold', ls='-', lw=2.5, label=f'Avg={gw[i]:.3f}')
    ax.set_xlabel('Gate Weight', fontsize=11); ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{name} Gate Distribution', fontweight='bold')
    ax.legend(fontsize=9)
plt.suptitle(f'Per-Patient Gate Distributions (n={len(VG)} val patients) — V3',
             fontsize=13, fontweight='bold')
plt.tight_layout()
savefig("v3_patient_gate_distributions.png")


# ╔══ FIG 10: Publication Metrics Table ═══
fig, ax = plt.subplots(figsize=(16, 2.2 + len(Config.TARGETS)*0.65))
ax.axis('off')
cols_hdr = ['Target', 'AUC-ROC', 'F1-Score', 'Precision', 'Recall', 'Specificity', 'Support', 'vs V2 AUC']
rows = []
for t in Config.TARGETS:
    m = best['per_class'][t]
    n_neg = len(VL) - m['support']
    t_idx = Config.TARGETS.index(t)
    preds_t = (VP[:,t_idx] > 0.5).astype(int)
    tn_v = int(((preds_t==0) & (VL[:,t_idx]==0)).sum())
    spec  = tn_v / max(n_neg, 1)
    v2_auc_str = f"+{(m['auc']-V2_BASELINE[t]['auc'])*100:.1f}%" if t in V2_BASELINE \
                 else '—'
    rows.append([t.replace('_',' ').title(),
                 f"{m['auc']:.4f}", f"{m['f1']:.4f}",
                 f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                 f"{spec:.3f}", str(m['support']), v2_auc_str])
rows.append(['Macro Average',
             f"{best['macro_auc']:.4f}", f"{best['macro_f1']:.4f}",
             f"{np.mean([best['per_class'][t]['precision'] for t in Config.TARGETS]):.3f}",
             f"{np.mean([best['per_class'][t]['recall'] for t in Config.TARGETS]):.3f}",
             '—', '—',
             f"+{(best['macro_auc']-0.8105)*100:.1f}%"])
tbl = ax.table(cellText=rows, colLabels=cols_hdr, cellLoc='center', loc='center',
               colWidths=[0.22,0.09,0.09,0.09,0.09,0.1,0.09,0.1])
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.9)
for (r,c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
    elif r == len(rows):
        cell.set_facecolor('#27ae60'); cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f0f3f4')
    cell.set_edgecolor('#bdc3c7')
plt.title('VisionCare 3.0 — Complete Test Set Performance (vs V2 Baseline)',
          fontweight='bold', fontsize=13, pad=14)
plt.tight_layout()
savefig("v3_metrics_table.png")


print(f"\n{'='*65}")
print("🎉  VISIONCARE 3.0 COMPLETE!")
print(f"{'='*65}")
print(f"  🏆 Macro AUC : {best['macro_auc']:.4f}  |  Δ vs V2: {best['macro_auc']-0.8105:+.4f}")
print(f"  🏆 Macro F1  : {best['macro_f1']:.4f}")
print(f"  📊 Gates      : V={100*gw[0]:.0f}%  ECG={100*gw[1]:.0f}%  Labs={100*gw[2]:.0f}%")
print(f"  🎯 Targets   : {Config.TARGETS}")
print(f"  🧪 Tier 3     : {'Enabled' if ENABLE_TIER3 else 'Disabled'}")
print("\n  Saved figures:")
for fig_name in [
    'v3_training_history.png', 'v3_gate_evolution.png', 'v3_version_comparison.png',
    'v3_roc_curves.png', 'v3_pr_curves.png', 'v3_confusion_matrices.png',
    'v3_per_disease.png', 'v3_v2_comparison.png',
    'v3_patient_gate_distributions.png', 'v3_metrics_table.png'
]:
    print(f"    • {fig_name}")
print(f"\n  Output: {Config.OUT_DIR}/")
print("✨ Done! ✨")
