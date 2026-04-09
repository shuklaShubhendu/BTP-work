# 🧠 VisionCare Model Architecture Guide

> **Complete documentation of all models used in the VisionCare multi-modal CVD detection system**

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Single vs Multi-Modal](#single-vs-multi-modal)
3. [Vision Models (CXR)](#-vision-models-chest-x-ray)
4. [Signal Models (ECG)](#-signal-models-ecg)
5. [Clinical Models (Labs)](#-clinical-models-blood-labs)
6. [Fusion Architecture](#-fusion-architecture)
7. [Model Selection Process](#model-selection-process)
8. [References](#references)

---

## Overview

VisionCare uses a **multi-modal deep learning approach** to predict cardiovascular disease risk. Instead of relying on a single data type, we combine:

| Modality | Data Type | Medical Significance |
|----------|-----------|---------------------|
| 🩻 **Vision** | Chest X-Ray | Shows heart size, lung congestion |
| ❤️ **Signal** | 12-Lead ECG | Reveals heart rhythm, electrical activity |
| 🩸 **Clinical** | Blood Labs | Indicates cholesterol, troponin, glucose |

By fusing information from all three sources, we achieve **higher accuracy** than any single modality alone.

---

## Single vs Multi-Modal

### What is a "Single Model"?

A **single-modality model** uses only ONE type of data:

```
┌───────────────────────────────────────────────────────────────────┐
│                    SINGLE MODALITY APPROACH                       │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   🩻 Vision Model                                                 │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │
│   │  Chest      │  →   │  DenseNet   │  →   │ CVD Risk:   │      │
│   │  X-Ray      │      │  or other   │      │ 67%         │      │
│   └─────────────┘      └─────────────┘      └─────────────┘      │
│                                                                   │
│   ❤️ Signal Model                                                 │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │
│   │  ECG        │  →   │  1D-CNN     │  →   │ CVD Risk:   │      │
│   │  Waveform   │      │  or other   │      │ 58%         │      │
│   └─────────────┘      └─────────────┘      └─────────────┘      │
│                                                                   │
│   🩸 Clinical Model                                               │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │
│   │  Blood      │  →   │  MLP        │  →   │ CVD Risk:   │      │
│   │  Labs       │      │  or TabNet  │      │ 52%         │      │
│   └─────────────┘      └─────────────┘      └─────────────┘      │
│                                                                   │
│   ⚠️ Problem: Each model misses information from other sources!  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### What is "Multi-Modal Fusion"?

A **fusion model** combines ALL data types:

```
┌───────────────────────────────────────────────────────────────────┐
│                    MULTI-MODAL FUSION APPROACH                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Same Patient Data (Hospital Admission #12345)                   │
│                                                                   │
│   🩻 Chest X-Ray ──────┐                                          │
│      (Enlarged heart)  │                                          │
│                        │      ┌─────────────┐                     │
│   ❤️ ECG Waveform ──────┼──→  │   FUSION    │  →  CVD Risk: 82%  │
│      (Arrhythmia)      │      │   MODEL     │     ✅ Higher!      │
│                        │      └─────────────┘                     │
│   🩸 Blood Labs ────────┘                                          │
│      (High troponin)                                              │
│                                                                   │
│   ✓ Captures complementary information!                          │
│   ✓ More accurate than any single source!                        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🩻 Vision Models (Chest X-Ray)

We compare **3 state-of-the-art** image classification architectures:

### 1. DenseNet-121 (Baseline)

```
Architecture:
┌─────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│Image│ → │ Dense Block │ → │ Dense Block │ → │ Dense Block │ → Features
└─────┘   │     1       │   │     2       │   │     3       │   (1024)
          └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                 │                 │                 │
                 └─────── Dense Connections ─────────┘
                    (Each layer connects to ALL previous layers)
```

| Property | Value |
|----------|-------|
| **Parameters** | 8 Million |
| **ImageNet Accuracy** | 74.4% |
| **Why Used** | Stanford's CheXNet used it for chest X-rays |
| **Key Feature** | Dense connections for feature reuse |

**Why DenseNet for Medical Imaging?**
- Dense connections allow **feature reuse** across layers
- Fewer parameters than ResNet-50 (8M vs 25M)
- Proven on CheXpert, NIH ChestX-ray14 datasets

---

### 2. EfficientNet-B2 (Efficient)

```
Architecture:
┌─────┐   ┌──────────────────────────────────┐
│Image│ → │     Compound Scaling             │ → Features
└─────┘   │  ┌───────────────────────────┐   │   (1408)
          │  │ Width × Depth × Resolution│   │
          │  │    All scaled together    │   │
          │  └───────────────────────────┘   │
          └──────────────────────────────────┘
```

| Property | Value |
|----------|-------|
| **Parameters** | 9 Million |
| **ImageNet Accuracy** | 80.1% |
| **Why Used** | Best accuracy per parameter |
| **Key Feature** | Compound scaling (width + depth + resolution) |

**Why EfficientNet?**
- **Compound scaling** balances network width, depth, and input resolution
- More efficient than manually designed networks
- B2 variant is optimal for ~20K sample datasets

---

### 3. ConvNeXt-Tiny (Modern)

```
Architecture:
┌─────┐   ┌────────────────────────────────────────┐
│Image│ → │        ConvNeXt Block                  │ → Features
└─────┘   │  ┌────────────────────────────────┐    │   (768)
          │  │  7×7 Depthwise Conv            │    │
          │  │  LayerNorm (instead of BN)     │    │
          │  │  GELU activation               │    │
          │  │  Inverted bottleneck           │    │
          │  └────────────────────────────────┘    │
          └────────────────────────────────────────┘
```

| Property | Value |
|----------|-------|
| **Parameters** | 28 Million |
| **ImageNet Accuracy** | 82.1% |
| **Why Used** | Modern CNN that competes with Vision Transformers |
| **Key Feature** | Incorporates transformer design principles into CNN |

**Why ConvNeXt?**
- Published in 2022 as "A ConvNet for the 2020s"
- Takes the best ideas from Vision Transformers but uses convolutions
- Often beats ViT on smaller datasets (no need for huge data)

---

### Vision Model Comparison

```
                    Accuracy vs Parameters (Vision)
    
    AUC-ROC │
    0.70    │                              ● ConvNeXt-Tiny
            │                                 (0.680)
    0.68    │
            │               ● EfficientNet-B2
    0.66    │                  (0.664)
            │   ● DenseNet-121
    0.64    │      (0.656)
            │
    0.62    │
            └─────────────────────────────────────────
                5M     10M     15M     20M     25M     30M
                              Parameters
```

---

## ❤️ Signal Models (ECG)

We compare **3 architectures** for time-series classification:

### 1. 1D-CNN (Baseline)

```
Architecture:
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
│12-Lead │ → │Conv 1D │ → │Conv 1D │ → │Conv 1D │ → Features
│  ECG   │   │ Pool   │   │ Pool   │   │ Pool   │   (256)
│5000×12 │   │64 ch   │   │128 ch  │   │256 ch  │
└────────┘   └────────┘   └────────┘   └────────┘
```

| Property | Value |
|----------|-------|
| **Parameters** | 0.5 Million |
| **Architecture** | 3 Convolutional layers + MaxPool |
| **Why Used** | Fast, simple baseline |
| **Key Feature** | Local pattern detection |

---

### 2. ResNet-1D (Deep)

```
Architecture:
┌────────┐   ┌─────────────────────────────────────┐
│12-Lead │ → │      Residual Blocks                │ → Features
│  ECG   │   │  ┌─────┐  ┌─────┐  ┌─────┐         │   (256)
│        │   │  │Block│→ │Block│→ │Block│         │
└────────┘   │  │ +   │  │ +   │  │ +   │         │
             │  └──┬──┘  └──┬──┘  └──┬──┘         │
             │     └────Skip Connections────┘      │
             └─────────────────────────────────────┘
```

| Property | Value |
|----------|-------|
| **Parameters** | 2 Million |
| **Architecture** | ResNet-18 adapted for 1D signals |
| **Why Used** | Skip connections for deeper learning |
| **Key Feature** | Can learn subtle patterns without gradient vanishing |

**Why Residual Connections for ECG?**
- ECG has subtle patterns (ST-segment changes, T-wave inversions)
- Deeper networks can capture these, but suffer from vanishing gradients
- Skip connections solve this problem

---

### 3. InceptionTime (SOTA)

```
Architecture:
┌────────┐   ┌───────────────────────────────────────────────┐
│12-Lead │ → │           Inception Module                    │
│  ECG   │   │  ┌────────────────────────────────────────┐   │
│        │   │  │  ┌────────┐ ┌────────┐ ┌────────┐     │   │
└────────┘   │  │  │Conv 1×1│ │Conv 3×1│ │Conv 7×1│     │   │ → Features
             │  │  └────────┘ └────────┘ └────────┘     │   │   (256)
             │  │       │          │          │         │   │
             │  │       └──── Concatenate ────┘         │   │
             │  │           (Multi-scale features)       │   │
             │  └────────────────────────────────────────┘   │
             └───────────────────────────────────────────────┘
```

| Property | Value |
|----------|-------|
| **Parameters** | 1.5 Million |
| **Architecture** | Parallel convolutions with different kernel sizes |
| **Why Used** | State-of-the-art for time-series classification |
| **Key Feature** | Captures both short beats and long rhythm patterns |

**Why InceptionTime for ECG?**
- ECG has **multi-scale patterns**:
  - Short: QRS complex (~100ms)
  - Medium: PR interval, QT interval (~200-400ms)
  - Long: Heart rate variability, rhythm (seconds)
- Parallel kernels (1, 3, 7, 15) capture all scales simultaneously

---

## 🩸 Clinical Models (Blood Labs)

### 1. MLP (Simple)

```
Architecture:
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│100 Labs │ → │ Dense   │ → │ Dense   │ → │ Dense   │ → 2 classes
│Features │   │ 128     │   │ 64      │   │ 64      │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
```

| Property | Value |
|----------|-------|
| **Parameters** | 20K (0.02M) |
| **Architecture** | 2 hidden layers |
| **Why Used** | Simple, effective for tabular data |

---

### 2. TabNet (Attention-Based)

```
Architecture:
┌─────────┐   ┌───────────────────────────────────────────┐
│100 Labs │ → │         Attention Mechanism               │
│Features │   │  Step 1: Select important features        │
└─────────┘   │  Step 2: Process selected features        │
              │  Step 3: Repeat, refine selection         │
              │                                           │
              │  "Focus on Troponin, ignore Hemoglobin"   │
              └───────────────────────────────────────────┘
                              ↓
                        2 classes
```

| Property | Value |
|----------|-------|
| **Parameters** | 100K (0.1M) |
| **Architecture** | Sequential attention with feature selection |
| **Why Used** | Interpretable - shows which labs matter |

**Why TabNet for Labs?**
- Clinical data has **varying importance** per patient
- For patient A: Troponin matters most
- For patient B: Glucose matters most
- TabNet learns to focus on relevant features

---

## 🔀 Fusion Architecture

### Intermediate Fusion Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VISIONCARE FUSION MODEL                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   🩻 VISION ENCODER (Best from comparison)                              │
│   ┌─────────────┐                                                       │
│   │  Chest      │ → [ConvNeXt-Tiny] ───────────→ 768 features          │
│   │  X-Ray      │                                     │                 │
│   └─────────────┘                                     │                 │
│                                                       │                 │
│   ❤️ SIGNAL ENCODER (Best from comparison)            │                 │
│   ┌─────────────┐                                     │                 │
│   │  12-Lead    │ → [InceptionTime] ───────────→ 256 features          │
│   │  ECG        │   or best winner        │           │                 │
│   └─────────────┘                         │           │                 │
│                                           │           │                 │
│   🩸 CLINICAL ENCODER (Best from comparison)          │                 │
│   ┌─────────────┐                         │           │                 │
│   │  Blood      │ → [MLP or TabNet] ───────────→ 64 features           │
│   │  Labs       │                         │           │                 │
│   └─────────────┘                         │           │                 │
│                                           │           │                 │
│   FUSION LAYERS                           │           │                 │
│   ┌───────────────────────────────────────┴───────────┴──────────────┐  │
│   │                                                                  │  │
│   │   Concatenate: [768 + 256 + 64] = 1088 features                 │  │
│   │                        ↓                                         │  │
│   │   Dense(512) → BatchNorm → ReLU → Dropout(0.4)                  │  │
│   │                        ↓                                         │  │
│   │   Dense(128) → ReLU → Dropout(0.2)                              │  │
│   │                        ↓                                         │  │
│   │   Dense(2) → Softmax                                            │  │
│   │                        ↓                                         │  │
│   │   [CVD Risk: 78%] [No CVD: 22%]                                 │  │
│   │                                                                  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Intermediate Fusion?

| Fusion Type | Method | Problem |
|-------------|--------|---------|
| **Early** | Concatenate raw inputs | Different data types don't mix |
| **Late** | Average prediction probabilities | Loses cross-modal relationships |
| **Intermediate** ✅ | Concatenate learned features | Best of both! |

**Intermediate fusion allows the model to learn patterns like:**
> "When CXR shows enlarged heart AND ECG shows arrhythmia AND labs show high troponin → Very high CVD risk"

---

## Model Selection Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MODEL SELECTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   STEP 1: Train all vision models                                       │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │ DenseNet-121  →  AUC: 0.656                            │           │
│   │ EfficientNet  →  AUC: 0.664                            │           │
│   │ ConvNeXt-Tiny →  AUC: 0.680  ✅ WINNER                 │           │
│   └─────────────────────────────────────────────────────────┘           │
│                                                                         │
│   STEP 2: Train all signal models                                       │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │ 1D-CNN       →  AUC: ?                                 │           │
│   │ ResNet-1D    →  AUC: ?                                 │           │
│   │ InceptionTime→  AUC: ?  → WINNER                       │           │
│   └─────────────────────────────────────────────────────────┘           │
│                                                                         │
│   STEP 3: Train all clinical models                                     │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │ MLP          →  AUC: ?                                 │           │
│   │ TabNet       →  AUC: ?  → WINNER                       │           │
│   └─────────────────────────────────────────────────────────┘           │
│                                                                         │
│   STEP 4: Combine winners into FUSION model                             │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │ ConvNeXt + InceptionTime + MLP/TabNet = FUSION         │           │
│   │ Expected AUC: 0.72 - 0.78  🎯                          │           │
│   └─────────────────────────────────────────────────────────┘           │
│                                                                         │
│   STEP 5: Evaluate on TEST set (held-out, never seen)                  │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │ Final unbiased performance for BTP presentation        │           │
│   └─────────────────────────────────────────────────────────┘           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## References

### Vision Models
1. **DenseNet**: Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
2. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling", ICML 2019
3. **ConvNeXt**: Liu et al., "A ConvNet for the 2020s", CVPR 2022

### Signal Models
4. **ResNet-1D**: Hannun et al., "Cardiologist-level arrhythmia detection", Nature Medicine 2019
5. **InceptionTime**: Fawaz et al., "InceptionTime: Finding AlexNet for Time Series", Data Mining 2020

### Clinical Models
6. **TabNet**: Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning", AAAI 2021

### Medical Imaging
7. **CheXNet**: Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection", arXiv 2017
8. **MIMIC-IV**: Johnson et al., "MIMIC-IV", Scientific Data 2023

---

<p align="center">
  <b>🫀 VisionCare - Multi-Modal Cardiovascular Disease Detection</b><br>
  <i>BTP Semester 7 • 2026</i>
</p>
