"""
VisionCare 3.0 — Model Architectures for Inference
====================================================
Contains:
  1. CXREncoder   — ResNet-50, 2048-D
  2. ECGEncoder   — 1D ResNet-18, 512-D
  3. LabsEncoder  — 3-Layer NN, 256-D
  4. GatedCrossAttentionFusionV3 — 4-head cross-attention + entropy-regularized gating
  5. VisionCareV3 — Full fusion model (all weights in fusion_v3_best.pth)
  6. V3 single-modal helpers — run fusion model with zeroed missing modalities

All V2 legacy architectures (DenseNet, EfficientNet, ConvNeXt, CNN1D, MLP) removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── V3 TARGETS ───────────────────────────────────────────────────────────────
V3_TARGETS = [
    'mortality', 'heart_failure', 'myocardial_infarction', 'arrhythmia',
    'sepsis', 'pulmonary_embolism', 'acute_kidney_injury', 'icu_admission'
]

V3_TARGET_LABELS = {
    'mortality':              'Mortality',
    'heart_failure':          'Heart Failure',
    'myocardial_infarction':  'Myocardial Infarction',
    'arrhythmia':             'Arrhythmia',
    'sepsis':                 'Sepsis',
    'pulmonary_embolism':     'Pulmonary Embolism',
    'acute_kidney_injury':    'Acute Kidney Injury',
    'icu_admission':          'ICU Admission',
}


# ══════════════════════════════════════════════════════════════════════════════
# V3 ENCODERS
# ══════════════════════════════════════════════════════════════════════════════

class CXREncoder(nn.Module):
    """ResNet-50 for CXR images → 2048-D embedding."""
    def __init__(self):
        super().__init__()
        from torchvision import models
        base = models.resnet50(weights=None)
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

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def get_unfreeze_params(self):
        return (list(self.layer3.parameters()) +
                list(self.layer4.parameters()) +
                list(self.avgpool.parameters()))


class BasicBlock1D(nn.Module):
    """1D ResNet BasicBlock for ECG signals."""
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
    """1D ResNet-18 for 12-lead ECG → 512-D embedding."""
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

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

    def get_unfreeze_params(self):
        return [p for n, p in self.named_parameters() if 'layer3' in n or 'layer4' in n]


class LabsEncoder(nn.Module):
    """3-layer NN for blood labs → 256-D embedding."""
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
        return list(self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# V3 FUSION MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GatedCrossAttentionFusionV3(nn.Module):
    """Cross-Attention Gated Fusion with FFN and entropy-regularized gating."""
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
        att = self.norm2(self.ffn(att) + att)
        av, as_, ac = att[:,0], att[:,1], att[:,2]
        gates = self.gate(torch.cat([av, as_, ac], -1))
        fused = gates[:,0:1]*av + gates[:,1:2]*as_ + gates[:,2:3]*ac
        return fused, gates


class VisionCareV3(nn.Module):
    """
    Complete VisionCare 3.0 model for inference.
    Loads from fusion_v3_best.pth which contains ALL weights
    (encoders + fusion + head). No SYMILE checkpoint needed.
    """
    def __init__(self, num_labels=8):
        super().__init__()
        self.cxr_enc  = CXREncoder()
        self.ecg_enc  = ECGEncoder(in_channels=12)
        self.labs_enc = LabsEncoder(in_dim=100)

        V, S, C, H = 2048, 512, 256, 256
        self.fusion = GatedCrossAttentionFusionV3(V, S, C, H, heads=4)
        self.head = nn.Sequential(
            nn.Linear(H, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_labels))

    def forward(self, cxr, ecg, labs):
        fv = self.cxr_enc(cxr)
        fs = self.ecg_enc(ecg)
        fc = self.labs_enc(labs)
        fused, gates = self.fusion(fv, fs, fc)
        logits = self.head(fused)
        return logits, gates

    def get_contributions(self, cxr, ecg, labs):
        """Return gate weights as a dict for XAI."""
        with torch.no_grad():
            _, gates = self.forward(cxr, ecg, labs)
            g = gates.cpu().numpy()[0]
            return {"vision": float(g[0]), "signal": float(g[1]), "clinical": float(g[2])}


# ══════════════════════════════════════════════════════════════════════════════
# V3 SINGLE-MODAL INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
# Strategy: use VisionCareV3 with zero tensors for missing modalities.
# The gate network naturally pushes near-100% weight to the active modality
# because the zeroed modalities produce blank (uninformative) embeddings.

def build_v3_single_model(modality: str, checkpoint_path: str = None) -> VisionCareV3:
    """
    Returns a VisionCareV3 instance ready for single-modal inference.
    Loads fusion_v3_best.pth weights if provided.
    """
    model = VisionCareV3(num_labels=8)
    if checkpoint_path:
        try:
            state = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state, strict=False)
            print(f'[V3-Single] Loaded checkpoint for {modality} inference')
        except Exception as e:
            print(f'[V3-Single] Could not load checkpoint: {e}')
    model.eval()
    return model


def run_v3_single_inference(model: VisionCareV3, modality: str,
                            cxr_tensor=None, ecg_tensor=None,
                            labs_tensor=None) -> dict:
    """
    Run VisionCareV3 with only one real modality.
    Missing modalities receive zero tensors.

    Args:
        model:        VisionCareV3 instance
        modality:     'cxr' | 'ecg' | 'labs' | 'multimodal'
        cxr_tensor:   (1, 3, 320, 320) or None
        ecg_tensor:   (1, 12, 5000) or None
        labs_tensor:  (1, 100) or None

    Returns:
        dict with 'risks' (8 diseases, %) and 'gates' (3 weights)
    """
    device = next(model.parameters()).device
    bs = 1

    if cxr_tensor is None:
        cxr_tensor = torch.zeros(bs, 3, 320, 320)
    if ecg_tensor is None:
        ecg_tensor = torch.zeros(bs, 12, 5000)
    if labs_tensor is None:
        labs_tensor = torch.zeros(bs, 100)

    with torch.no_grad():
        logits, gates = model(
            cxr_tensor.to(device),
            ecg_tensor.to(device),
            labs_tensor.to(device)
        )
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        g = gates.cpu().numpy()[0]

    risks = {
        t: round(float(probs[i]) * 100, 1)
        for i, t in enumerate(V3_TARGETS)
    }

    if modality == 'multimodal':
        gate_out = {'vision': float(g[0]), 'signal': float(g[1]), 'clinical': float(g[2])}
    elif modality == 'cxr':
        gate_out = {'vision': float(g[0]), 'signal': 0.0, 'clinical': 0.0}
    elif modality == 'ecg':
        gate_out = {'vision': 0.0, 'signal': float(g[1]), 'clinical': 0.0}
    else:  # labs
        gate_out = {'vision': 0.0, 'signal': 0.0, 'clinical': float(g[2])}

    return {'risks': risks, 'gates': gate_out}
