"""
VisionCare — Single-Modal Model Architectures
=============================================
Defines all single-encoder model classes used for CXR-only, ECG-only,
and Labs-only inference.  Kept separate so main.py can import without
triggering any Colab/training-side imports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── CXR MODELS (input: B×3×H×W, output: (logits_6, features)) ──────────────

class DenseNet121MultiLabel(nn.Module):
    name = "DenseNet-121"
    feature_dim = 1024

    def __init__(self, num_labels=6):
        super().__init__()
        from torchvision import models
        self.backbone = models.densenet121(weights=None)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, num_labels))

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features), features


class EfficientNetB2MultiLabel(nn.Module):
    name = "EfficientNet-B2"
    feature_dim = 1408

    def __init__(self, num_labels=6):
        super().__init__()
        from torchvision import models
        self.backbone = models.efficientnet_b2(weights=None)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1408, num_labels))

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features), features


class ConvNeXtTinyMultiLabel(nn.Module):
    name = "ConvNeXt-Tiny"
    feature_dim = 768

    def __init__(self, num_labels=6):
        super().__init__()
        from torchvision import models
        self.backbone = models.convnext_tiny(weights=None)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Flatten(1), nn.LayerNorm(768), nn.Dropout(0.3), nn.Linear(768, num_labels)
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.mean(dim=[-2, -1])
        return self.classifier(features), features


CXR_ARCHITECTURES = [DenseNet121MultiLabel, EfficientNetB2MultiLabel, ConvNeXtTinyMultiLabel]


# ─── ECG MODELS (input: B×12×seq_len, output: (logits_6, features)) ─────────

class CNN1DMultiLabel(nn.Module):
    name = "1D-CNN"
    feature_dim = 256

    def __init__(self, num_labels=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 11, padding=5),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 7, padding=3),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_labels))

    def forward(self, x):
        features = self.conv(x).squeeze(-1)
        return self.classifier(features), features


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1DMultiLabel(nn.Module):
    name = "ResNet-1D"
    feature_dim = 256

    def __init__(self, num_labels=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, 15, stride=2, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResBlock1D(64,  64),  ResBlock1D(64,  64))
        self.layer2 = nn.Sequential(ResBlock1D(64,  128, stride=2), ResBlock1D(128, 128))
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


ECG_ARCHITECTURES = [CNN1DMultiLabel, ResNet1DMultiLabel]


# ─── LABS MODELS (input: B×100, output: (logits_6, features)) ────────────────
# Input is 50 lab-percentile features + 50 missingness flags (from MIMIC preprocessing).

class MLPMultiLabel(nn.Module):
    name = "MLP"
    feature_dim = 64

    def __init__(self, input_dim=100, num_labels=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, num_labels))

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features), features


LABS_ARCHITECTURES = [MLPMultiLabel]


# ─── LABEL → HF/MORTALITY MAPPING ────────────────────────────────────────────
# Single models output 6 pathology labels:
#   [Cardiomegaly, Edema, Atelectasis, Pleural Effusion, Lung Opacity, No Finding]
# These map to HF/mortality using the same proxy used in colab_fusion_v2.py:
#   heart_failure proxy = Edema OR (Cardiomegaly AND Pleural Effusion)

LABEL_NAMES = ['Cardiomegaly', 'Edema', 'Atelectasis', 'Pleural Effusion', 'Lung Opacity', 'No Finding']
HF_WEIGHTS  = [0.30, 0.40, 0.05, 0.20, 0.05, 0.00]   # weighted combination


def map_6labels_to_risks(probs_6: list) -> dict:
    """
    Convert 6-label sigmoid probabilities to hf_risk / mortality_risk (0-100).
    Uses clinical weighting aligned with colab_fusion_v2 proxy logic.
    """
    cardiomegaly  = probs_6[0]
    edema         = probs_6[1]
    pleural_eff   = probs_6[3]
    no_finding    = probs_6[5]

    # Primary proxy: edema OR (cardiomegaly AND pleural effusion), same as v2 proxy
    hf_proxy = max(edema, cardiomegaly * pleural_eff)
    # Weighted fallback blended in for stability
    hf_weighted = sum(w * p for w, p in zip(HF_WEIGHTS, probs_6))
    hf_score = 0.6 * hf_proxy + 0.4 * hf_weighted
    # No-finding suppresses risk
    hf_score = hf_score * (1.0 - 0.5 * no_finding)

    hf_risk       = round(min(hf_score * 100, 97.0), 1)
    mortality_risk = round(min(hf_score * 0.42 * 100, 90.0), 1)

    return {
        "hf_risk":        hf_risk,
        "mortality_risk": mortality_risk,
        "pathology_probs": {name: round(float(p), 3) for name, p in zip(LABEL_NAMES, probs_6)},
    }
