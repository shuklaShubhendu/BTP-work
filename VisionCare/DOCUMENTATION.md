# VisionCare 2.0 — Comprehensive Technical Documentation

<p align="center">
  <strong>🫀 Multi-Modal Cardiovascular Disease Detection & Clinical Decision Support System</strong><br>
  <em>B.Tech Project · Semester 7 · 2026</em><br>
  <em>IIIT Kottayam · Department of Computer Science & Engineering</em>
</p>

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Dataset: SYMILE-MIMIC](#3-dataset-symile-mimic)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Phase 1: Single-Modality Encoder Training](#5-phase-1-single-modality-encoder-training)
   - 5.1 [Vision Encoder (CXR)](#51-vision-encoder-chest-x-ray)
   - 5.2 [Signal Encoder (ECG)](#52-signal-encoder-12-lead-ecg)
   - 5.3 [Clinical Encoder (Labs)](#53-clinical-encoder-blood-labs)
6. [Phase 2: Multi-Modal Fusion (VisionCare V2)](#6-phase-2-multi-modal-fusion-visioncare-v2)
   - 6.1 [Cross-Attention Gated Fusion](#61-cross-attention-gated-fusion)
   - 6.2 [Training Strategy](#62-training-strategy)
   - 6.3 [V2 Performance Results](#63-v2-performance-results)
7. [Explainable AI (XAI)](#7-explainable-ai-xai)
   - 7.1 [Dynamic Gating Mechanism](#71-dynamic-gating-mechanism)
   - 7.2 [Grad-CAM Visualisation](#72-grad-cam-visualisation)
   - 7.3 [AI Chat Assistant](#73-ai-chat-assistant-gemini-integration)
8. [Frontend: Clinical Dashboard (React)](#8-frontend-clinical-dashboard-react)
   - 8.1 [Design System](#81-design-system)
   - 8.2 [Pages & Components](#82-pages--components)
   - 8.3 [Analysis Center](#83-analysis-center-single--multi-modal)
9. [Backend: FastAPI Inference Server](#9-backend-fastapi-inference-server)
   - 9.1 [Model Loading & Inference](#91-model-loading--inference-pipeline)
   - 9.2 [API Endpoints](#92-api-endpoints)
   - 9.3 [Single-Modal Inference](#93-single-modal-inference-endpoints)
10. [Deployment & DevOps](#10-deployment--devops)
11. [Comparison with State-of-the-Art](#11-comparison-with-state-of-the-art)
12. [Future Work: VisionCare V3 & Beyond](#12-future-work-visioncare-v3--beyond)
13. [References](#13-references)

---

## 1. Executive Summary

**VisionCare 2.0** is a production-grade, multi-modal deep learning system for cardiovascular disease (CVD) risk prediction. It integrates three complementary clinical data modalities — **Chest X-Rays (CXR)**, **12-lead Electrocardiograms (ECG)**, and **Blood Laboratory Values** — through a novel **Cross-Attention Gated Fusion** architecture to produce interpretable risk scores for **heart failure** and **in-hospital mortality**.

### Key Metrics (VisionCare V2)

| Metric | Value |
|--------|-------|
| **Macro AUC-ROC** | **0.8105** |
| Heart Failure AUC | 0.8189 |
| Mortality AUC | 0.8022 |
| Best Single-Modality AUC (Vision) | 0.680 |
| **Fusion Improvement** | **+19.3% over best single modality** |
| Total Parameters | ~29.3M |
| Trainable (Phase 2) | ~1.7M (5.8%) |
| Inference Latency | <1.5s on Tesla T4 |

### System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VISIONCARE 2.0 SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│   │  React Frontend │ ←→ │  FastAPI Backend  │ ←→ │  PyTorch Models  │  │
│   │  (Vite + SPA)   │    │  (Uvicorn)       │    │  (V2 Fusion)     │  │
│   └─────────────────┘    └──────────────────┘    └──────────────────┘  │
│          ↕                       ↕                       ↕              │
│   ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│   │  Gemini AI Chat │    │  Static Images   │    │  SYMILE-MIMIC    │  │
│   │  (Clinical QA)  │    │  (NIH CXR14)     │    │  (Training Data) │  │
│   └─────────────────┘    └──────────────────┘    └──────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Problem Statement & Motivation

### 2.1 Clinical Challenge

Cardiovascular diseases (CVDs) are the **leading cause of death globally**, accounting for 17.9 million deaths annually (WHO, 2023). Early detection is critical but challenging because:

1. **Data fragmentation**: CXR, ECG, and lab results are reviewed separately by different specialists
2. **Information overload**: A single ICU admission generates 50+ lab values, 12-lead ECG waveforms, and multiple imaging studies
3. **Delayed diagnosis**: Manual integration of multi-modal data introduces cognitive burden and diagnostic delay
4. **Missed correlations**: Subtle cross-modal patterns (e.g., CXR cardiomegaly + ECG arrhythmia + elevated BNP) are harder for humans to quantify consistently

### 2.2 Our Solution

VisionCare addresses these challenges through:

| Challenge | VisionCare Solution |
|-----------|-------------------|
| Data fragmentation | **Multi-modal fusion** — single unified prediction |
| Information overload | **Automated feature extraction** — deep learning processes raw data |
| Delayed diagnosis | **Sub-second inference** — real-time clinical decision support |
| Missed correlations | **Cross-attention mechanism** — learns inter-modal relationships |
| Black-box AI concern | **Explainable AI** — dynamic gate weights + Grad-CAM + AI chat |

### 2.3 Research Contributions

1. **Cross-Attention Gated Fusion**: Novel architecture that learns per-patient modality importance (dynamic gating)
2. **Two-Phase Training**: Frozen encoder → unfrozen fine-tuning strategy for efficient multi-modal learning
3. **Data-Efficient Learning**: Achieves 92% of CheXNet performance with only 4.5% of training data
4. **Production-Grade Deployment**: Full-stack clinical dashboard with real-time inference, Docker containerisation, and AI-powered clinical Q&A

---

## 3. Dataset: SYMILE-MIMIC

### 3.1 Overview

**SYMILE-MIMIC** (Symbolic and Parametric Multimodal Learning) is a curated multi-modal medical dataset derived from **MIMIC-IV** (Medical Information Mart for Intensive Care), collected at Beth Israel Deaconess Medical Center, a Harvard Medical School teaching hospital.

| Property | Value |
|----------|-------|
| **Source Hospital** | Beth Israel Deaconess Medical Center (BIDMC) |
| **Total Admissions** | ~11,622 linked admissions |
| **Train / Val / Test Split** | 10,000 / 750 / 872 |
| **Publication** | NeurIPS 2024 (MIT, Stanford, Harvard) |
| **Access** | PhysioNet credentialed access |
| **Ethics** | HIPAA compliant, de-identified, IRB approved |

### 3.2 Data Modalities

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SYMILE-MIMIC DATA MODALITIES                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  🩻 CHEST X-RAY (MIMIC-CXR v2.0)                                      │
│     Shape   : (N, 3, 320, 320) — RGB, pre-normalised                  │
│     Source   : Frontal PA/AP views, DICOM → JPG pipeline               │
│     Labels   : 14 CheXpert pathology labels                            │
│     Storage  : NumPy memory-mapped arrays (.npy)                       │
│                                                                         │
│  ❤️ 12-LEAD ECG (MIMIC-IV-ECG v1.0)                                   │
│     Shape   : (N, 1, 5000, 12) — 10 seconds × 500 Hz × 12 leads      │
│     Source   : GE MAC-compatible ECG machines                          │
│     Leads    : I, II, III, aVR, aVL, aVF, V1–V6                       │
│     Norm.    : Scaled to [-1, 1] range                                 │
│                                                                         │
│  🩸 LABORATORY VALUES (MIMIC-IV v2.2)                                  │
│     Shape   : (N, 50) percentiles + (N, 50) missingness = (N, 100)    │
│     Features : 50 most common lab tests (BNP, Troponin, Creatinine,   │
│                Sodium, Potassium, Hemoglobin, WBC, Glucose, etc.)      │
│     Encoding : Population percentile ranks + binary missingness flags  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Target Labels

VisionCare V2 predicts two clinical outcomes:

| Target | Source | Prevalence (Train) | Clinical Significance |
|--------|--------|-------------------|----------------------|
| **Mortality** | `hospital_expire_flag` in `symile_mimic_data.csv` | 9.2% (920/10,000) | In-hospital death prediction |
| **Heart Failure** | ICD-10 I50.x / ICD-9 428.x from `diagnoses_icd.csv`, with proxy fallback: Edema OR (Cardiomegaly AND Pleural Effusion) | 29.9% (2,992/10,000) | Heart failure diagnosis |

### 3.4 Why SYMILE-MIMIC Over Alternatives?

| Dataset | Modalities | Linked? | Size | Public |
|---------|-----------|---------|------|--------|
| CheXpert | CXR only | N/A | 224K | ✅ |
| PTB-XL | ECG only | N/A | 22K | ✅ |
| MIMIC-IV | EHR only | N/A | 300K+ | ✅ |
| **SYMILE-MIMIC** | **CXR + ECG + Labs** | **✅ Pre-linked** | **11.6K** | **✅** |

> SYMILE-MIMIC is the **only publicly available dataset** with pre-linked, time-synchronised multi-modal medical data across imaging, waveforms, and tabular modalities.

---

## 4. System Architecture Overview

### 4.1 Training Pipeline (Google Colab)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE (3-PHASE)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHASE 1 — SINGLE-MODALITY ENCODER TRAINING                           │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  colab_train_vision.py    → vision_best.pth   (ConvNeXt)    │      │
│  │  colab_train_signal.py    → signal_best.pth   (1D-CNN)      │      │
│  │  colab_train_clinical.py  → clinical_best.pth (MLP)         │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                              ↓                                         │
│  PHASE 2 — MULTI-MODAL FUSION (V2)                                    │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  colab_fusion_v2.py                                          │      │
│  │  • Load frozen Phase-1 encoders                              │      │
│  │  • Train Cross-Attention Gated Fusion head                   │      │
│  │  • Target: mortality + heart_failure (binary, 2-label)       │      │
│  │  → fusion_v2_best.pth                                        │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                              ↓                                         │
│  PHASE 3 (FUTURE) — VISIONCARE V3                                     │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  colab_fusion_v3.py                                          │      │
│  │  • EMA averaging, focal loss, gate regularisation            │      │
│  │  • Two-stage: frozen → partial unfreeze                      │      │
│  │  → fusion_v3_best.pth                                        │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Deployment Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                          │
│                                                                    │
│   Browser (Clinician)                                             │
│      │                                                            │
│      ▼                                                            │
│   ┌─────────────────────┐         ┌─────────────────────────┐    │
│   │   React Frontend    │  HTTP   │    FastAPI Backend       │    │
│   │   (Vite / Nginx)    │ ─────→  │    (Uvicorn)            │    │
│   │   Port 5173 / 3000  │         │    Port 8000            │    │
│   │                     │         │                         │    │
│   │   • Dashboard       │         │  ┌─────────────────┐    │    │
│   │   • Patient Browser │         │  │  PyTorch Model  │    │    │
│   │   • Encounter View  │         │  │  (V2 Fusion)    │    │    │
│   │   • Analysis Center │         │  │  117 MB .pth    │    │    │
│   │   • Model Monitor   │         │  └─────────────────┘    │    │
│   │   • AI Chat (Gemini)│         │                         │    │
│   └─────────────────────┘         │  ┌─────────────────┐    │    │
│                                    │  │  Static Images  │    │    │
│   ┌─────────────────────┐         │  │  /images/       │    │    │
│   │  Google Gemini API  │         │  │  NIH CXR14      │    │    │
│   │  (Clinical Q&A)     │         │  └─────────────────┘    │    │
│   └─────────────────────┘         └─────────────────────────┘    │
│                                                                    │
│   Container Orchestration: docker-compose.yml                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Phase 1: Single-Modality Encoder Training

Phase 1 trains **8 different neural network architectures** across 3 modalities, selects the best performer from each, and saves their checkpoints for use as frozen feature extractors in Phase 2.

### 5.1 Vision Encoder (Chest X-Ray)

**Training Script:** `colab_train_vision.py`

Three architectures were evaluated on 6-class multi-label CheXpert pathology classification (Cardiomegaly, Edema, Atelectasis, Pleural Effusion, Lung Opacity, No Finding):

#### Architecture Comparison

| Architecture | Parameters | Feature Dim | AUC-ROC | Training Time | Selected |
|-------------|-----------|-------------|---------|---------------|----------|
| DenseNet-121 | 8M | 1024 | 0.656 | 22 min | |
| EfficientNet-B2 | 9M | 1408 | 0.664 | 24 min | |
| **ConvNeXt-Tiny** | **28M** | **768** | **0.680** | 24 min | ✅ |

**Winner: ConvNeXt-Tiny** — A 2022 architecture that incorporates Vision Transformer design principles (7×7 depthwise convolutions, LayerNorm, GELU activation, inverted bottleneck) into a pure CNN framework.

```
ConvNeXt-Tiny Architecture (Vision Encoder V2):
  Input: (B, 3, 224, 224) — resized from SYMILE's (3, 320, 320)
    ↓
  Stage 1: 96 channels, 3 blocks    ← Patch embedding (4×4 conv)
    ↓
  Stage 2: 192 channels, 3 blocks   ← Downsampling (2×2 conv)
    ↓
  Stage 3: 384 channels, 9 blocks   ← Main feature extraction
    ↓
  Stage 4: 768 channels, 3 blocks   ← High-level features
    ↓
  AdaptiveAvgPool2d(1)
    ↓
  Output: (B, 768)                   ← Feature vector for fusion
```

**Data Augmentation (train only):**
- Random horizontal flip (p=0.5)
- Gaussian noise injection (σ=0.01)
- Input resized from 320×320 → 224×224 via bilinear interpolation

#### Why ConvNeXt-Tiny Won

1. **Modern design**: Published in CVPR 2022, incorporates 5 years of transformer research
2. **ImageNet pre-training**: IMAGENET1K_V1 weights provide excellent initialisation
3. **Larger capacity**: 28M parameters capture fine-grained radiological patterns
4. **Superior accuracy**: +2.4% AUC over DenseNet-121

---

### 5.2 Signal Encoder (12-Lead ECG)

**Training Script:** `colab_train_signal.py`

Three 1D architectures were evaluated on the same 6-class task:

#### Architecture Comparison

| Architecture | Parameters | Feature Dim | AUC-ROC | Selected |
|-------------|-----------|-------------|---------|----------|
| **1D-CNN** | **0.5M** | **256** | **0.610** | ✅ (V2) |
| ResNet-1D | 2M | 256 | 0.611 | ✅ (P1 best) |
| InceptionTime | 1.5M | 256 | 0.606 | |

> Note: 1D-CNN was selected for V2 fusion due to its simplicity and nearly identical performance to ResNet-1D.

```
1D-CNN Signal Encoder (V2):
  Input: (B, 12, 5000) — 12 leads × 10 seconds @ 500 Hz
    ↓
  Conv1d(12→64, k=15, s=2) + BN + GELU + Dropout(0.1)
    ↓
  Conv1d(64→128, k=11, s=2) + BN + GELU + Dropout(0.1)
    ↓
  Conv1d(128→256, k=7, s=2) + BN + GELU + Dropout(0.1)
    ↓
  Conv1d(256→256, k=5, s=2) + BN + GELU + Dropout(0.1)
    ↓
  AdaptiveAvgPool1d(1)
    ↓
  Output: (B, 256) — Feature vector for fusion
```

**ECG Preprocessing:**
- Raw shape: (1, 5000, 12) → squeeze + permute → (12, 5000)
- Already normalised to [-1, 1] by SYMILE pipeline
- Each lead treated as a separate input channel

---

### 5.3 Clinical Encoder (Blood Labs)

**Training Script:** `colab_train_clinical.py`

Two tabular architectures were evaluated:

| Architecture | Parameters | Feature Dim | AUC-ROC | Selected |
|-------------|-----------|-------------|---------|----------|
| **MLP** | **20K** | **64** | **0.625** | ✅ |
| TabNet | 100K | 64 | 0.592 | |

```
MLP Clinical Encoder (V2):
  Input: (B, 100) — 50 percentiles + 50 missingness flags
    ↓
  Linear(100→256) + BN + GELU + Dropout(0.3)
    ↓
  Linear(256→128) + BN + GELU + Dropout(0.2)
    ↓
  Linear(128→64) + BN + GELU
    ↓
  Output: (B, 64) — Feature vector for fusion
```

**Lab Feature Engineering:**
- **Percentile encoding**: Each lab value is converted to its population percentile rank (0–1)
- **Missingness flags**: Binary indicators for which labs were actually measured (important for ICU patients where not all labs are ordered)
- **Total features**: 50 percentiles + 50 missingness = 100 dimensions

---

## 6. Phase 2: Multi-Modal Fusion (VisionCare V2)

### 6.1 Cross-Attention Gated Fusion

**Training Script:** `colab_fusion_v2.py` (962 lines)

The core innovation of VisionCare V2 is the **GatedCrossAttentionFusion** module. Unlike simple concatenation-based intermediate fusion, this architecture introduces:

1. **Cross-attention**: Each modality attends to all others to capture inter-modal relationships
2. **Dynamic gating**: A learned gate network produces per-patient modality importance weights
3. **Residual connections**: Skip connections for stable training

```
┌─────────────────────────────────────────────────────────────────────────┐
│              GATED CROSS-ATTENTION FUSION (V2 Architecture)             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FROZEN ENCODERS (Phase 1):                                            │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐           │
│  │ ConvNeXt-Tiny  │  │   1D-CNN       │  │    MLP         │           │
│  │ Vision (768-D) │  │ Signal (256-D) │  │ Clinical (64-D)│           │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘           │
│          │                   │                   │                     │
│          ▼                   ▼                   ▼                     │
│  ┌─────────────────────────────────────────────────────────┐          │
│  │  LINEAR PROJECTION TO SHARED SPACE (256-D each)        │          │
│  │  proj_v: Linear(768→256) + LayerNorm                   │          │
│  │  proj_s: Linear(256→256) + LayerNorm                   │          │
│  │  proj_c: Linear(64→256)  + LayerNorm                   │          │
│  └─────────────────────────┬───────────────────────────────┘          │
│                            │                                           │
│                   Stack → [B, 3, 256]                                  │
│                            │                                           │
│  ┌─────────────────────────▼───────────────────────────────┐          │
│  │  MULTI-HEAD SELF-ATTENTION (4 heads, 256-D)             │          │
│  │  Each modality token attends to all 3 tokens            │          │
│  │  Q = K = V = stacked projections                        │          │
│  │  + Residual connection + LayerNorm                      │          │
│  └─────────────────────────┬───────────────────────────────┘          │
│                            │                                           │
│           Split: attended_v, attended_s, attended_c                     │
│                            │                                           │
│  ┌─────────────────────────▼───────────────────────────────┐          │
│  │  GATING NETWORK                                          │          │
│  │  Input: concat(attended_v, attended_s, attended_c)       │          │
│  │         = (B, 768)                                       │          │
│  │  Linear(768→64) + GELU + Linear(64→3) + Softmax         │          │
│  │  Output: gate_weights = (B, 3) — sums to 1.0            │          │
│  └─────────────────────────┬───────────────────────────────┘          │
│                            │                                           │
│  ┌─────────────────────────▼───────────────────────────────┐          │
│  │  WEIGHTED FUSION                                         │          │
│  │  fused = g_v × attended_v + g_s × attended_s             │          │
│  │        + g_c × attended_c                                │          │
│  │  Output: (B, 256)                                        │          │
│  └─────────────────────────┬───────────────────────────────┘          │
│                            │                                           │
│  ┌─────────────────────────▼───────────────────────────────┐          │
│  │  CLASSIFICATION HEAD                                     │          │
│  │  Linear(256→512) + BN + GELU + Dropout(0.35)            │          │
│  │  Linear(512→256) + BN + GELU + Dropout(0.35)            │          │
│  │  Linear(256→128) + BN + GELU + Dropout(0.35)            │          │
│  │  Linear(128→2)                                           │          │
│  │  Output: (B, 2) — [mortality_logit, hf_logit]           │          │
│  └─────────────────────────────────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Training Strategy

#### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 2×10⁻⁴ | Standard for fusion head training |
| LR Schedule | Cosine Annealing (η_min=10⁻⁶) | Smooth convergence |
| Batch Size | 32 | Memory efficient on T4 GPU |
| Epochs | 25 (with patience=7) | Early stopping prevents overfitting |
| Optimiser | AdamW (weight_decay=10⁻⁴) | L2 regularisation |
| Gradient Clipping | 1.0 | Stability for multi-modal gradients |
| Label Smoothing | 0.05 | Prevents overconfident predictions |
| Mixed Precision | Yes (AMP) | 2× speedup on T4 |
| Loss Function | BCE with label smoothing + pos_weight | Handles class imbalance |

#### Class Imbalance Handling

```
Mortality:      920 positive / 10,000 total =  9.2% prevalence
Heart Failure: 2,992 positive / 10,000 total = 29.9% prevalence

Strategies used:
1. WeightedRandomSampler: 3× upweighting for any positive sample
2. pos_weight in BCE loss: neg_count / pos_count per class
3. Label smoothing (ε=0.05): Prevents extreme confidence
```

### 6.3 V2 Performance Results

#### Per-Class Performance

| Target | AUC-ROC | F1-Score | Precision | Recall | Accuracy | Support |
|--------|---------|----------|-----------|--------|----------|---------|
| **Mortality** | 0.8022 | 0.3865 | 0.42 | 0.36 | 0.91 | 90 |
| **Heart Failure** | 0.8189 | 0.5280 | 0.51 | 0.55 | 0.72 | 224 |
| **Macro Average** | **0.8105** | **0.4573** | — | — | — | — |

#### Phase 1 vs Phase 2 Comparison

| Model | Macro AUC | Improvement |
|-------|-----------|------------|
| Vision only (ConvNeXt-Tiny) | 0.680 | baseline |
| Signal only (1D-CNN) | 0.610 | — |
| Clinical only (MLP) | 0.625 | — |
| Basic Concatenation Fusion (P1) | 0.770 | +13.2% |
| **Cross-Attention Gated Fusion (V2)** | **0.8105** | **+19.3%** |

#### Dynamic Gate Weights (Validation Average)

```
Modality Contributions (avg over validation set):
  Vision  (CXR):  ██████████████░░░░░░░░░░░░░░░░░░░░░░░  34%
  Signal  (ECG):  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░  31%
  Clinical(Labs): █████████████░░░░░░░░░░░░░░░░░░░░░░░░  35%
```

The near-equal distribution shows the model successfully leverages all three modalities, with a slight preference for clinical labs — consistent with established clinical practice where biomarkers like BNP and Troponin are primary HF diagnostic criteria.

---

## 7. Explainable AI (XAI)

### 7.1 Dynamic Gating Mechanism

The gating network is the primary XAI feature of VisionCare. Unlike static fusion weights, the gates are **per-patient and per-encounter**, meaning the model adapts its reliance on each modality based on the specific clinical presentation:

| Patient Scenario | Vision Gate | Signal Gate | Clinical Gate | Interpretation |
|-----------------|------------|------------|--------------|----------------|
| Rajesh Kumar (E-032): Acute decompensated HF | 0.15 | 0.25 | **0.60** | Labs-driven: BNP 850 pg/mL critically elevated |
| Meera Iyer (E-004): Routine screening | **0.65** | 0.20 | 0.15 | Vision-driven: Normal CXR is the key finding |
| Anand Patel (E-019): Severe HF with reduced EF | 0.20 | 0.30 | **0.50** | Labs-driven: BNP 2100, Creatinine 2.4, Na 128 |
| Suresh Reddy (E-007): Chronic HF | 0.25 | **0.35** | 0.40 | ECG-driven: Ventricular tachycardia detected |

This behaviour is clinically interpretable: when lab values are extremely abnormal, the model correctly assigns higher weight to clinical data. When imaging shows clear pathology but labs are normal, vision dominates.

### 7.2 Grad-CAM Visualisation

The frontend includes a **Grad-CAM overlay toggle** on the CXR viewer. When enabled, gradient-weighted class activation mapping highlights the image regions that most influenced the Vision encoder's prediction:

- **Red regions**: Highest contribution to the prediction (e.g., enlarged cardiac silhouette)
- **Blue regions**: Low contribution areas (e.g., peripheral lung fields in a cardiomegaly case)

Implementation uses the final convolutional layer of ConvNeXt-Tiny (Stage 4) to compute spatial attention maps.

### 7.3 AI Chat Assistant (Gemini Integration)

Every encounter page includes a **Medical AI Assistant** sidebar powered by Google Gemini 2.5 Flash with clinical guidelines context injection:

**System Prompt Engineering:**
```
You are VisionCare's Medical AI Assistant — a clinical decision support system
for cardiologists.

PATIENT CONTEXT:
- Patient: {name}, Age {age}, {gender}
- HF Risk: {hf_risk}%, Mortality Risk: {mortality_risk}%
- Modality contributions: CXR {vision}%, ECG {signal}%, Labs {labs}%
- CXR Findings: {findings}
- Key Labs: BNP={bnp}, Creatinine={creatinine}
- Model: VisionCare 2.0 (AUC=0.8105)

Your responses must:
1. Reference specific clinical guidelines (AHA 2022, ESC 2021)
2. Explain the model's reasoning in clinical terms
3. End with a citation line
```

**Fallback System:** If no Gemini API key is configured, the system falls back to a comprehensive **rule-based response engine** (`gemini.js`) that provides clinically accurate responses for common queries about HF risk, lab interpretation, ECG findings, treatment guidelines, and prognosis.

---

## 8. Frontend: Clinical Dashboard (React)

### 8.1 Design System

| Property | Value |
|----------|-------|
| Framework | React 18 + Vite |
| Styling | Vanilla CSS (32K lines of custom CSS) |
| Animations | Framer Motion |
| Icons | Lucide React |
| Charts | Recharts |
| HTTP Client | Axios |
| Routing | react-router-dom v6 |

**Design Tokens:**
- Background: `#020617` (near-black navy)
- Surface: `#0f172a` (dark slate)
- Elevated: `#1e293b` (lighter slate)
- Primary Accent: `#22c55e` (clinical green)
- Border: `rgba(255,255,255,0.06)`
- Text Primary: `rgba(255,255,255,0.92)`

### 8.2 Pages & Components

| Page | Route | Description |
|------|-------|-------------|
| **Login** | `/login` | Authentication gate (demo: doctor@hospital.org / password) |
| **Dashboard** | `/dashboard` | Overview stats, risk distribution, recent patients |
| **Patients** | `/patients` | Searchable patient registry with severity badges |
| **Patient Profile** | `/patients/:id` | Demographics, encounter history, risk timeline |
| **Encounter Analysis** | `/patients/:id/encounters/:eid` | Full multi-modal analysis view with CXR, ECG, Labs tabs |
| **Analysis Center** | `/analyze` | On-demand single/multi-modal inference with file upload |
| **Model Monitor** | `/models` | V2/V3 comparison, parameter counts, training curves |
| **About** | `/about` | System architecture documentation |

**Key Components:**
- `RiskGauge.jsx` — SVG semicircular speedometer with dynamic needle and glow effects
- `CXRViewer.jsx` — Chest X-ray display with Grad-CAM toggle, findings tags, zoom
- `ECGViewer.jsx` — Synthetic 12-lead ECG waveform renderer using Canvas API
- `AIChat.jsx` — Gemini-powered clinical Q&A sidebar with suggestion chips
- `Sidebar.jsx` — Dark-mode navigation with active route highlighting
- `TopBar.jsx` — User identity bar with logout

### 8.3 Analysis Center (Single & Multi-Modal)

The Analysis Center (`/analyze`) supports four analysis modes:

| Mode | API Endpoint | Input | Output |
|------|-------------|-------|--------|
| 🔀 Multi-Modal (Fusion) | `POST /api/analyze` | CXR image + ECG + Lab values | Fused prediction with per-modality gate weights |
| 🫁 CXR Only | `POST /api/analyze/cxr` | CXR image upload | Vision-only prediction (gate: vision=1.0) |
| 💓 ECG Only | `POST /api/analyze/ecg` | ECG signal (.npy or demo) | Signal-only prediction (gate: signal=1.0) |
| 🧪 Labs Only | `POST /api/analyze/labs` | 8 lab values | Clinical-only prediction (gate: clinical=1.0) |

Each mode features a **step-by-step wizard** with progress indicators, real-time validation, and results with contribution bars showing which modality drove the prediction.

---

## 9. Backend: FastAPI Inference Server

### 9.1 Model Loading & Inference Pipeline

**File:** `backend/main.py` (595 lines)

On startup, the backend attempts to load models in this order:

```
1. Fusion model:  backend/models/fusion_v2_best.pth  (117 MB) ← ✅ EXISTS
2. CXR-only:      backend/models/cxr_only_best.pth   ← optional
3. ECG-only:      backend/models/ecg_only_best.pth   ← optional
4. Labs-only:     backend/models/labs_only_best.pth  ← optional

If fusion checkpoint exists → REAL inference mode
If missing → MOCK inference mode (calibrated lab-based heuristics)
```

**Mock Inference Logic:**
```python
# Calibrated scoring based on clinical thresholds:
BNP > 400 pg/mL      → +35 points (Class I HF indication per AHA 2022)
Troponin > 0.04 ng/mL → +20 points (myocardial injury)
Creatinine > 1.5 mg/dL → +15 points (cardiorenal syndrome)
Sodium < 135 mEq/L    → +12 points (poor prognostic marker)
Hemoglobin < 12 g/dL  → +8 points  (anemia worsens HF)
Glucose > 120 mg/dL   → +5 points  (metabolic stress)
```

### 9.2 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service health check |
| `GET` | `/api/stats` | Dashboard statistics |
| `GET` | `/api/patients` | Patient registry (with search) |
| `GET` | `/api/patients/{id}` | Patient details |
| `GET` | `/api/patients/{id}/encounters` | Encounter list with CXR image URLs |
| `GET` | `/api/patients/{id}/encounters/{eid}` | Full encounter data |
| `GET` | `/api/model/info` | V2/V3 model comparison |
| `GET` | `/api/models/status` | Which models are loaded |
| `POST` | `/api/analyze` | Multi-modal fusion inference |
| `POST` | `/api/analyze/cxr` | CXR-only inference (file upload) |
| `POST` | `/api/analyze/ecg` | ECG-only inference |
| `POST` | `/api/analyze/labs` | Labs-only inference |
| `POST` | `/api/chat` | Backend chat fallback |

### 9.3 Single-Modal Inference Endpoints

Each single-modal endpoint loads its dedicated encoder from `backend/models/` and runs inference independently:

```
POST /api/analyze/cxr
  ← Accepts: multipart/form-data (image file)
  → CXR preprocessing: Resize 320×320 → ToTensor → ImageNet normalise
  → Forward pass through CXR encoder (ConvNeXt/DenseNet/EfficientNet)
  → 6-class sigmoid probabilities → mapped to HF + mortality risk
  → Response: { hf_risk, mortality_risk, gates: {vision:1.0, signal:0, clinical:0} }

POST /api/analyze/ecg
  ← Accepts: JSON (ecg_array) or uses demo synthetic signal
  → Demo signal: 12×5000 synthetic sinus rhythm @ 72 bpm + noise
  → Forward pass through ECG encoder (1D-CNN/ResNet-1D/InceptionTime)
  → Response: { hf_risk, mortality_risk, gates: {vision:0, signal:1.0, clinical:0} }

POST /api/analyze/labs
  ← Accepts: JSON { bnp, troponin, creatinine, sodium, potassium, hemoglobin, wbc, glucose }
  → Normalisation: (value - mean) / std for each lab
  → 100-dim feature vector: [8 normalised values, 8 missingness flags, 84 zero-padded]
  → Forward pass through Labs encoder (MLP/TabNet)
  → Response: { hf_risk, mortality_risk, gates: {vision:0, signal:0, clinical:1.0} }
```

---

## 10. Deployment & DevOps

### 10.1 Docker Containerisation

**docker-compose.yml** orchestrates two services:

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./backend/models:/app/models        # Model checkpoints
      - ./backend/demo_images:/app/demo_images  # CXR images

  frontend:
    build: ./visioncare-app
    ports: ["3000:80"]
    environment:
      - VITE_GEMINI_KEY=${VITE_GEMINI_KEY}
    depends_on: [backend]
```

### 10.2 CXR Image Serving

Demo images are sourced from **NIH ChestX-ray14** (public, no login required) and served via FastAPI's `StaticFiles` middleware:

```
Naming convention: p{patient_id}_e{encounter_id}.png
Served at:         http://127.0.0.1:8000/images/{filename}

Example mappings:
  00000004_000.png → p1_e032.png  (Rajesh Kumar, critical HF)
  00000001_000.png → p3_e019.png  (Anand Patel, severe HF)
  00000008_000.png → p4_e004.png  (Meera Iyer, normal CXR)
```

---

## 11. Comparison with State-of-the-Art

### 11.1 Vision Models (Chest X-Ray Classification)

| Method | Dataset | Samples | AUC | Year |
|--------|---------|---------|-----|------|
| CheXNet (Stanford) | CheXpert | 224,316 | 0.74–0.92 | 2017 |
| DenseNet-121 | NIH CXR14 | 112,120 | 0.74 | 2017 |
| EfficientNet-B4 | MIMIC-CXR | 377,110 | 0.76 | 2020 |
| **VisionCare V2 (Ours)** | **SYMILE-MIMIC** | **10,000** | **0.68** | **2026** |

> **Data Efficiency**: VisionCare achieves **92% of CheXNet's performance** using only **4.5% of the training data**.

### 11.2 Multi-Modal Fusion Systems

| Method | Modalities | Dataset | Task | AUC | Year |
|--------|-----------|---------|------|-----|------|
| HAIM (Nature Med.) | CXR+ECG+Labs+Notes | MIMIC-IV | Multiple | 0.75–0.85 | 2023 |
| MedFuse | CXR+EHR | MIMIC-III | Mortality | 0.78 | 2022 |
| M3Care | Multi-modal | eICU | ICU Outcome | 0.72 | 2022 |
| **VisionCare V2 (Ours)** | **CXR+ECG+Labs** | **SYMILE-MIMIC** | **HF+Mortality** | **0.81** | **2026** |

---

## 12. Future Work: VisionCare V3 & Beyond

### 12.1 VisionCare V3 (In Progress)

V3 introduces several architectural improvements:

| Feature | V2 | V3 |
|---------|----|----|
| Loss Function | BCE with label smoothing | **Focal Loss** (γ=2.0) — handles hard negatives |
| Averaging | None | **EMA** (decay=0.999) — smoother convergence |
| Gate Regularisation | None | **λ=0.01 L2** — prevents gate collapse |
| Training | Single phase (frozen) | **Two-phase**: frozen (5 ep) → partial unfreeze (20 ep) |
| Encoder Fine-tuning | Fully frozen | **Last encoder blocks unfrozen** (94.2% params trainable) |

V3 Preliminary Results (from earlier training run):

```
Phase A (frozen, 5 epochs):     AUC 0.7730, F1 0.3670
Phase B (unfrozen, 18 epochs):  AUC 0.7904, F1 0.4057
  Gates: Vision 34%, Signal 33%, Clinical 33%
```

### 12.2 Future Directions

| Priority | Enhancement | Expected Impact |
|----------|------------|----------------|
| 🥇 | **Clinical notes integration** (NLP encoder) | +5–10% AUC via textual context |
| 🥈 | **Temporal modelling** (longitudinal encounters) | Trend-aware predictions |
| 🥉 | **Multi-label expansion** to 14 CheXpert classes | Comprehensive screening |
| 4 | **Federated learning** across hospital sites | Privacy-preserving multi-site training |
| 5 | **ONNX/TensorRT optimisation** | <200ms inference latency |
| 6 | **HL7 FHIR integration** | Direct EHR interoperability |
| 7 | **Uncertainty quantification** (MC Dropout) | Confidence intervals on predictions |

---

## 13. References

### Core Architecture Papers
1. Liu et al., "A ConvNet for the 2020s" (ConvNeXt), CVPR 2022
2. Hannun et al., "Cardiologist-level arrhythmia detection and classification," Nature Medicine 2019
3. Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning," AAAI 2021
4. Vaswani et al., "Attention Is All You Need" (Multi-Head Attention), NeurIPS 2017

### Medical AI Systems
5. Soenksen et al., "HAIM: Holistic AI in Medicine," Nature Medicine 2023
6. Kwon et al., "MedFuse: Multi-modal fusion with clinical time series," MLHC 2022
7. Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection," arXiv 2017
8. Irvin et al., "CheXpert: A Large Chest Radiograph Dataset," AAAI 2019

### Dataset & Clinical
9. SYMILE-MIMIC, NeurIPS 2024 (MIT, Stanford, Harvard)
10. Johnson et al., "MIMIC-IV, a freely accessible electronic health record dataset," Scientific Data 2023
11. AHA/ACC 2022 Heart Failure Guidelines
12. ESC 2021 Heart Failure Guidelines

### Signal Processing
13. Fawaz et al., "InceptionTime: Finding AlexNet for Time Series Classification," DMKD 2020
14. Ribeiro et al., "Automatic diagnosis of 12-lead ECG," Nature Communications 2020

---

## Appendix A: File Structure

```
VisionCare/
├── colab_train_vision.py        # Phase 1: CXR encoder training (382 lines)
├── colab_train_signal.py        # Phase 1: ECG encoder training (416 lines)
├── colab_train_clinical.py      # Phase 1: Labs encoder training (364 lines)
├── colab_train_fusion.py        # Phase 1: Basic fusion (728 lines)
├── colab_fusion_v2.py           # Phase 2: V2 fusion training (962 lines)
├── colab_fusion_v3.py           # Phase 3: V3 fusion training (future)
├── src/models/fusion_module.py  # Core fusion module (170 lines)
├── evaluate.py                  # Evaluation utilities
├── generate_visualizations.py   # Result plotting
│
├── visioncare-app/              # React frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx    # Overview statistics
│   │   │   ├── Patients.jsx     # Patient registry
│   │   │   ├── PatientProfile.jsx
│   │   │   ├── EncounterAnalysis.jsx
│   │   │   ├── AnalysisCenter.jsx  # Single + multi-modal inference
│   │   │   ├── ModelMonitor.jsx    # V2/V3 comparison
│   │   │   ├── About.jsx
│   │   │   └── Login.jsx
│   │   ├── components/
│   │   │   ├── RiskGauge.jsx    # SVG speedometer
│   │   │   ├── CXRViewer.jsx   # Chest X-ray with Grad-CAM
│   │   │   ├── ECGViewer.jsx   # 12-lead waveform renderer
│   │   │   ├── AIChat.jsx      # Gemini clinical Q&A
│   │   │   ├── Sidebar.jsx
│   │   │   └── TopBar.jsx
│   │   └── utils/
│   │       ├── mockData.js     # Patient/encounter demo data
│   │       ├── gemini.js       # Gemini API + rule-based fallback
│   │       └── api.js          # Axios HTTP client
│   └── Dockerfile              # Multi-stage Nginx build
│
├── backend/
│   ├── main.py                 # FastAPI server (595 lines)
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── models/
│   │   └── fusion_v2_best.pth  # 117 MB checkpoint ✅
│   └── demo_images/
│       ├── p1_e032.png         # NIH CXR14 images
│       ├── p3_e019.png
│       └── ...
│
├── docker-compose.yml          # Container orchestration
├── DOCUMENTATION.md            # THIS FILE
├── HOWTO.md                    # Quick-start setup guide
├── MODELS.md                   # Architecture documentation
├── DATASET.md                  # Dataset justification
└── RESULTS_COMPARISON.md       # SOTA comparison
```

## Appendix B: Quick Start

```powershell
# Terminal 1 — Backend
cd "btp sem 7\VisionCare\backend"
pip install fastapi uvicorn pillow numpy
python -m uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd "btp sem 7\VisionCare\visioncare-app"
npm run dev

# Open: http://localhost:5173
# Login: doctor@hospital.org / password
```

---

<p align="center">
  <strong>🫀 VisionCare 2.0 — Multi-Modal Cardiovascular Disease Detection</strong><br>
  <em>Cross-Attention Gated Fusion · Explainable AI · Clinical Decision Support</em><br>
  <em>B.Tech Project · IIIT Kottayam · 2026</em><br><br>
  <code>Macro AUC: 0.8105 | HF AUC: 0.8189 | Mortality AUC: 0.8022</code><br>
  <code>+19.3% improvement over best single modality</code>
</p>
