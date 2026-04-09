# VisionCare 2.0: Multi-Modal Clinical Decision Support System
## Project Specification & Technical Architecture Document

---

**Document Version:** 2.0  
**Institution:** Indian Institute of Information Technology Kottayam  
**Program:** B.Tech. — Bachelor of Technology (Semester 7)  
**Project Category:** AI/ML Research & Engineering  
**Date:** March 2026  

---

## Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Motivation & Research Gap](#2-motivation--research-gap)  
3. [Objectives](#3-objectives)  
4. [Phase 1 Results Summary (VisionCare 1.0)](#4-phase-1-results-summary-visioncare-10)  
5. [Phase 2 Strategy: Why Fusion Will Win](#5-phase-2-strategy-why-fusion-will-win)  
6. [Dataset & Data Engineering](#6-dataset--data-engineering)  
7. [AI/ML Model Architecture](#7-aiml-model-architecture)  
8. [Full-Stack System Architecture](#8-full-stack-system-architecture)  
9. [Database Schema](#9-database-schema)  
10. [Backend API Specification (FastAPI)](#10-backend-api-specification-fastapi)  
11. [Frontend Architecture (React / Next.js)](#11-frontend-architecture-react--nextjs)  
12. [Explainable AI (XAI) & RAG Chatbot](#12-explainable-ai-xai--rag-chatbot)  
13. [Project Execution Timeline](#13-project-execution-timeline)  
14. [Expected Outcomes & Success Metrics](#14-expected-outcomes--success-metrics)  
15. [References](#15-references)  

---

## 1. Project Overview

**VisionCare** is a multi-modal, AI-powered Clinical Decision Support System (CDSS) designed to assist physicians in the diagnosis and risk stratification of complex cardiovascular diseases. The system fuses three distinct patient data modalities — Chest X-Rays (CXR), 12-Lead Electrocardiograms (ECG), and Laboratory Blood Tests — into a single, unified deep learning model capable of producing holistic, explainable disease predictions.

Unlike single-modality AI tools that rely solely on visual imaging, VisionCare is built on the clinical reality that complex diseases such as Heart Failure cannot be reliably diagnosed from a single source of information. A cardiologist requires imaging evidence of congestion, electrical evidence of cardiac stress, and biochemical confirmation from blood biomarkers before reaching a definitive conclusion. VisionCare replicates this multi-source reasoning pipeline using state-of-the-art deep learning.

VisionCare 2.0 is the second and final phase of this research project. It is characterized by:
- A strategic pivot in prediction targets to clinically meaningful **systemic outcomes** (Heart Failure, Mortality Risk).
- A complete full-stack product involving a **FastAPI backend**, a **React frontend dashboard**, and an **Explainable AI** layer comprising both a visual contribution analysis engine and a **RAG-powered medical LLM chatbot**.

---

## 2. Motivation & Research Gap

### 2.1 The Clinical Bottleneck
Cardiovascular disease (CVD) is the leading cause of death globally, accounting for an estimated 17.9 million deaths per year (WHO, 2023). Early and accurate diagnosis is the single greatest lever for reducing this toll. Yet clinical diagnosis of complex cardiac conditions remains fragmented: radiologists interpret CXRs, cardiologists review ECGs, and laboratory physicians analyze blood panels — often in isolation and with significant delays.

### 2.2 Limitations of Single-Modality AI
The majority of existing medical AI research focuses on a single data modality. CheXNet (Stanford, 2017) demonstrated that a DenseNet-121 could classify 14 chest pathologies from X-rays alone. However, for truly systemic conditions like Heart Failure, the X-ray captures only one dimension of the disease — pulmonary congestion — while completely missing electrical dysfunction and biochemical stress markers.

### 2.3 The Multi-Modal Imperative
The diagnostic standard for Heart Failure per the 2022 ACC/AHA Guidelines requires:
1. **Imaging:** Evidence of pulmonary congestion or cardiomegaly on CXR.
2. **Electrocardiography:** Detection of atrial fibrillation, ST-segment changes, or left ventricular hypertrophy voltage criteria.
3. **Biomarkers:** Elevated Natriuretic Peptide (BNP/NT-proBNP), Troponin, or Creatinine.

No single AI modality can capture all three dimensions. Multi-modal fusion is not a research curiosity — it is a clinical necessity.

---

## 3. Objectives

| # | Objective | Category |
|---|-----------|----------|
| 1 | Build a multi-modal deep learning model fusing CXR, ECG, and Lab data. | Core AI |
| 2 | Demonstrate statistically significant performance superiority of the fusion model over any single-modality baseline. | Research Validation |
| 3 | Target clinically meaningful outcomes (Heart Failure, In-Hospital Mortality). | Clinical Relevance |
| 4 | Develop a production-grade RESTful API for real-time inference. | Backend Engineering |
| 5 | Design an intuitive clinical dashboard with per-modality contribution visualization. | Frontend Engineering |
| 6 | Implement an Explainable AI layer including Grad-CAM, SHAP, and a RAG LLM chatbot. | Explainability & Trust |

---

## 4. Phase 1 Results Summary (VisionCare 1.0)

In Phase 1, VisionCare trained and evaluated 8 deep learning models across 3 modalities using the SYMILE-MIMIC dataset (10,000 patients, 6 disease labels). All models were trained on Google Colab using T4 GPUs.

### 4.1 Individual Model Performance
| Modality | Best Model | Macro-AUC | Accuracy | Parameters |
|----------|------------|-----------|----------|------------|
| Vision | ConvNeXt-Tiny 🏆 | 0.7694 | 74.3% | 28M |
| Vision | DenseNet-121 | 0.7531 | 73.2% | 8M |
| Vision | EfficientNet-B2 | 0.7512 | 73.9% | 9M |
| Signal | 1D-CNN 🏆 | 0.6022 | 69.4% | 0.5M |
| Signal | ResNet-1D | 0.5934 | 69.6% | 2M |
| Clinical | MLP 🏆 | 0.6118 | 69.6% | 20K |
| Clinical | TabNet | 0.5953 | 69.2% | 100K |

### 4.2 Multi-Modal Fusion Performance
| Configuration | Macro-AUC | Compared to Vision-Only |
|---------------|-----------|------------------------|
| Vision only | 0.7694 | — |
| Signal only | 0.6022 | -16.7% |
| Clinical only | 0.6118 | -15.8% |
| **Multi-Modal Fusion** | **0.7702** | **+0.08%** |

### 4.3 Key Finding & Limitation
The marginal improvement of the fusion model (+0.08%) over Vision alone revealed a fundamental design flaw: the 6 target diseases (Cardiomegaly, Pleural Effusion, Edema, Lung Opacity, Atelectasis, No Finding) are **radiographic findings** — defined by what is directly visible on an X-ray. For such targets, the Vision model possesses an inherent, insurmountable advantage over ECG and Lab data because Pleural Effusion, by definition, is a visual chest finding. ECG signals and laboratory values provide little additional diagnostic power for these specific labels.

This critical insight directly motivates the Phase 2 strategy.

---

## 5. Phase 2 Strategy: Why Fusion Will Win

### 5.1 The Paradigm Shift
The strategic pivot in VisionCare 2.0 is to replace radiographic-finding labels with **systemic clinical outcome labels**. For systemic outcomes, no single modality is sufficient, forcing the Fusion model to extract genuinely complementary information from all three data sources.

### 5.2 Target Disease Selection
| Disease | CXR Contribution | ECG Contribution | Labs Contribution | Why Fusion is Essential |
|---------|-----------------|-----------------|-------------------|------------------------|
| **Heart Failure (HF)** | Cardiomegaly, pulmonary congestion | LV strain, AF, ST changes | **BNP > 400 (gold standard)**, Troponin, Creatinine | No single test is definitive; all three together provide 95%+ sensitivity |
| **In-Hospital Mortality** | Severity of acute findings | Lethal arrhythmias, ischemic changes | Lactate, Organ failure markers | Survival prediction requires assessing the entire physiological system |
| **Acute Respiratory Failure** | Bilateral opacities | Tachycardia, right heart strain | PaO2/FiO2 ratio, ABG | Pulmonary failure manifests across all three modalities simultaneously |

### 5.3 Expected Performance Improvement
Based on published benchmarks from HAIM (Nature Medicine, 2023) and MedFuse (2022), transitioning to systemic clinical outcome targets is projected to yield the following:

| Modality | Projected AUC (HF Prediction) | Rationale |
|----------|-------------------------------|-----------|
| Vision only | ~0.65 | CXR finds congestion but misses electrical & biochemical evidence |
| Signal only | ~0.65 | ECG finds arrythmia but cannot "see" fluid accumulation |
| Clinical only | ~0.67 | BNP alone is strong but misses structural evidence |
| **Fusion (All 3)** | **~0.78 – 0.82** | All three modalities provide independent, complementary evidence |

**This represents a projected 13–17% AUC improvement of Fusion over any single modality — a decisive and demonstrable multi-modal win.**

### 5.4 Model Reuse & Transfer Learning Strategy
The pre-trained encoder models saved in Google Drive from Phase 1 will be partially reused to avoid redundant computation.

| Model Component | Phase 2 Strategy | Rationale |
|----------------|-----------------|-----------|
| ConvNeXt-Tiny Feature Extractor | ✅ **REUSE (Frozen)** | Already trained to extract lung/cardiac visual features |
| 1D-CNN Signal Encoder | ✅ **REUSE (Frozen)** | Already trained to read ECG waveform patterns |
| MLP Clinical Encoder | ✅ **REUSE (Frozen)** | Feature extraction from lab percentiles already learned |
| Final Classification Heads | 🔄 **RETRAIN (New Head)** | New heads for Heart Failure & Mortality labels |
| Multi-Modal Fusion MLP | 🔄 **RETRAIN (New)** | New fusion network for new target labels |

*Retraining only the final layers (< 5% of total parameters) reduces training time from hours to minutes.*

---

## 6. Dataset & Data Engineering

### 6.1 Primary Dataset: SYMILE-MIMIC
- **Source:** PhysioNet (Holste et al., 2024)
- **Cohort:** 10,000 patient admissions (train) + 750 (validation)
- **Modalities:** Chest X-Ray (224×224 JPEG), 12-Lead ECG (12×5000 waveform), Laboratory Values (100 features: 50 percentile + 50 missingness flags)

### 6.2 Label Augmentation: MIMIC-IV Integration
To generate new clinical outcome labels for the existing 10,000 SYMILE patients, we will cross-reference their `subject_id` index against the MIMIC-IV `core` and `hosp` modules.

| New Label | Source Table | Filter Condition |
|-----------|-------------|-----------------|
| Heart Failure | `diagnoses_icd` | ICD-10: I50.x; ICD-9: 428.x |
| In-Hospital Mortality | `admissions` | `deathtime IS NOT NULL` |
| Acute Resp. Failure | `diagnoses_icd` | ICD-10: J96.x; ICD-9: 518.81 |

**Data Pipeline:**
```
SYMILE-MIMIC 10K patients
      ↓ (Join on subject_id)
MIMIC-IV diagnoses_icd + admissions
      ↓ (Pandas merge + ICD code filter)
New Multi-Label Target CSV
      ↓ (Drop old CheXpert labels)
Updated PyTorch Dataset (Same CXR/ECG/Lab .npy arrays)
```
*The existing preprocessed `.npy` arrays are fully reused — only the label CSV changes.*

---

## 7. AI/ML Model Architecture

### 7.1 Individual Encoders
| Encoder | Architecture | Input | Output Embedding | Frozen in Phase 2 |
|---------|-------------|-------|-----------------|-------------------|
| **Vision** | ConvNeXt-Tiny (ImageNet pre-trained) | 224×224×3 CXR | 768-D vector | ✅ Yes |
| **Signal** | 1D-CNN (3 Conv layers) | 12×5000 ECG | 256-D vector | ✅ Yes |
| **Clinical** | MLP (100 → 256 → 64) | 100-D Lab vector | 64-D vector | ✅ Yes |

### 7.2 Fusion Model
```
Vision  Encoder  →  768D  ─┐
Signal  Encoder  →  256D  ─┼─→  [CONCAT: 1088D]  →  Fusion MLP  →  Output
Clinical Encoder →   64D  ─┘

Fusion MLP: Linear(1088→512) → BN → ReLU → Dropout(0.3)
            Linear(512→256)  → BN → ReLU → Dropout(0.3)
            Linear(256→128)  → BN → ReLU
            Linear(128→N)    → Sigmoid          [N = number of targets]
```

### 7.3 Explainability Layer
Immediately after inference, the system computes modality contribution scores using **Integrated Gradients** applied to the concatenation layer. This produces a percentage attribution for each modality:
```python
# Conceptual
contributions = integrated_gradients(
    baseline = zero_embedding,
    input    = [vision_emb, signal_emb, clinical_emb],
    target   = predicted_class
)
# Returns: {"vision": 15%, "signal": 25%, "clinical": 60%}
```

---

## 8. Full-Stack System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    REACT FRONTEND                       │
│   Patient Dashboard | XAI Visualizer | RAG Chatbot      │
└──────────────┬───────────────────────┬──────────────────┘
               │ HTTP / WebSocket      │ REST API
┌──────────────▼───────────────────────▼──────────────────┐
│                    FASTAPI BACKEND                       │
│   Auth | File Upload | Inference | XAI | RAG Endpoint   │
└────────┬─────────────────────┬──────────────┬───────────┘
         │                     │              │
┌────────▼───────┐   ┌─────────▼───────┐  ┌──▼────────────┐
│   PostgreSQL   │   │  PyTorch        │  │  ChromaDB /   │
│  Patient DB    │   │  Model Server   │  │  FAISS VectorDB│
│  (Predictions) │   │  (Inference)    │  │  (Med Guidelines)│
└────────────────┘   └─────────────────┘  └───────────────┘
```

### Tech Stack Summary
| Layer | Technology | Justification |
|-------|------------|---------------|
| Frontend | React + Next.js + Tailwind CSS | Component-based, SSR for fast initial load |
| Backend API | FastAPI (Python) | Async, native Python, ideal for ML serving |
| ML Framework | PyTorch | Best-in-class for research model serving |
| Primary Database | PostgreSQL + SQLAlchemy | Relational integrity for clinical records |
| Vector Database | ChromaDB | Lightweight, local, persistent for RAG |
| LLM | Llama 3 (local) or GPT-4o (API) | Medical reasoning for chatbot |

---

## 9. Database Schema

### 9.1 Relational Database (PostgreSQL)

```sql
-- Core entity for a hospital patient
CREATE TABLE patients (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mrn         VARCHAR(50) UNIQUE NOT NULL,  -- Medical Record Number
    age         INTEGER,
    gender      VARCHAR(10),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- A single hospital admission/visit
CREATE TABLE encounters (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id    UUID REFERENCES patients(id) ON DELETE CASCADE,
    admitted_at   TIMESTAMP NOT NULL,
    discharged_at TIMESTAMP,
    clinical_notes TEXT
);

-- The 3 uploaded modalities for an encounter
CREATE TABLE modality_data (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id  UUID REFERENCES encounters(id) ON DELETE CASCADE,
    type          VARCHAR(20) CHECK (type IN ('CXR', 'ECG', 'LABS')),
    file_path     TEXT NOT NULL,    -- Cloud Storage URI
    features_path TEXT,             -- Path to pre-computed .npy embedding
    uploaded_at   TIMESTAMP DEFAULT NOW()
);

-- AI-generated risk prediction for an encounter
CREATE TABLE predictions (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id            UUID REFERENCES encounters(id) ON DELETE CASCADE,
    hf_risk_score           FLOAT,       -- Heart Failure probability [0.0–1.0]
    mortality_risk_score    FLOAT,       -- Mortality probability [0.0–1.0]
    vision_contribution     FLOAT,       -- % contribution from CXR encoder
    signal_contribution     FLOAT,       -- % contribution from ECG encoder
    clinical_contribution   FLOAT,       -- % contribution from Labs encoder
    predicted_at            TIMESTAMP DEFAULT NOW()
);

-- LLM-generated textual explanation for a prediction
CREATE TABLE xai_explanations (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id         UUID REFERENCES predictions(id) ON DELETE CASCADE,
    llm_reasoning         TEXT,          -- Full RAG-generated explanation text
    gradcam_image_path    TEXT,          -- URI to Grad-CAM heatmap image
    created_at            TIMESTAMP DEFAULT NOW()
);

-- Audit log for all chatbot interactions
CREATE TABLE chatbot_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id    UUID REFERENCES encounters(id),
    user_query      TEXT NOT NULL,
    llm_response    TEXT NOT NULL,
    rag_sources     JSONB,             -- [{title: "", page: "", excerpt: ""}]
    created_at      TIMESTAMP DEFAULT NOW()
);
```

### 9.2 Vector Database Schema (ChromaDB)

The vector database stores chunked medical guidelines for the RAG chatbot.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique chunk identifier |
| `embedding` | float[] | 768-D text embedding (e.g., via `all-MiniLM-L6`) |
| `document` | string | Raw text chunk (≈ 500 tokens) |
| `metadata.source` | string | Document title (e.g., "AHA 2022 HF Guidelines") |
| `metadata.section` | string | Section heading |
| `metadata.page` | int | Page number for citation |
| `metadata.year` | int | Publication year |

**Ingested Medical Literature:**
- 2022 AHA/ACC Heart Failure Guidelines
- 2021 ESC Guidelines for Diagnosis of Heart Failure
- AHA Scientific Statement on AI in Healthcare
- 2023 Chest X-Ray AI Interpretation Guide (RSNA)
- Internal MIMIC-IV Clinical Notes Lexicon

---

## 10. Backend API Specification (FastAPI)

### 10.1 Authentication
All endpoints are protected by JWT Bearer token authentication.
```
POST /api/auth/login     → Returns JWT access token
POST /api/auth/refresh   → Refreshes expired token
```

### 10.2 Patient & Encounter Management
```
POST   /api/patients                      → Create new patient record
GET    /api/patients/{id}                 → Get patient profile
POST   /api/patients/{id}/encounters      → Create a new encounter
GET    /api/encounters/{encounter_id}     → Get encounter summary
```

### 10.3 Data Upload
```
POST /api/encounters/{encounter_id}/upload
    Content-Type: multipart/form-data
    Body:
        cxr_file: .png / .dcm (Chest X-Ray image)
        ecg_file: .npy / .csv (12×5000 ECG waveform)
        labs_file: .json / .csv (100 lab features)

    Response:
        {
          "modality_data_id": "...",
          "status": "uploaded",
          "features_extracted": true
        }
```

### 10.4 Inference Engine
```
POST /api/predict/{encounter_id}
    Body: {} (empty — uses uploaded modalities)

    Response:
        {
          "prediction_id": "...",
          "hf_risk_score": 0.82,
          "mortality_risk_score": 0.31,
          "modality_contributions": {
              "vision":   0.15,
              "signal":   0.25,
              "clinical": 0.60
          },
          "predicted_at": "2026-03-26T15:00:00Z"
        }
```

### 10.5 XAI Endpoints
```
GET /api/xai/{prediction_id}/gradcam
    Response: PNG image stream (heatmap overlaid on CXR)

GET /api/xai/{prediction_id}/contributions
    Response: {vision: 15%, signal: 25%, clinical: 60%}
```

### 10.6 RAG Chatbot Endpoint
```
POST /api/chat/explain
    Body:
        {
          "encounter_id": "...",
          "query": "Why did the AI heavily weight clinical labs?"
        }

    Response (Server-Sent Events / Streaming):
        data: {"token": "The", "done": false}
        data: {"token": "patient's", "done": false}
        ...
        data: {"token": ".", "done": true, "sources": ["AHA 2022 HF Guidelines, p.14"]}
```

---

## 11. Frontend Architecture (React / Next.js)

### 11.1 Page Structure
```
/app
├── (auth)
│   └── /login                      → Login page
├── /dashboard
│   └── /                           → Patient list & search
├── /patients
│   ├── /[id]                       → Patient profile
│   └── /[id]/encounters/[enc_id]  → Encounter detail view (main UI)
└── /about                          → System documentation
```

### 11.2 Encounter Detail View (The Core UI — Page `/patients/[id]/encounters/[enc_id]`)

This is the primary clinical interface, divided into 3 panels:

**Left Panel — Modality Viewer (40% width)**
- **CXR Tab:** Displays the chest X-ray with Grad-CAM heatmap overlay toggle.
- **ECG Tab:** Interactive 12-lead ECG waveform chart (Recharts library). Highlights abnormal segments.
- **Labs Tab:** Color-coded lab table. Values outside normal range are highlighted in red/orange.

**Center Panel — AI Risk Dashboard (30% width)**
- Large risk score dials (e.g., "82% Heart Failure Risk").
- **Modality Contribution Bar Chart:** Horizontal stacked bars showing the contribution of each encoder to the prediction:
  ```
  CXR    [████░░░░░░] 15%
  ECG    [████████░░] 25%
  Labs   [██████████████████████████] 60%
  ```
- Prediction timestamp and model version.

**Right Panel — Medical AI Chatbot (30% width)**
- Persistent chat window.
- Pre-loaded context with current prediction data.
- Suggested starter questions:
  - "Explain the Heart Failure risk for this patient."
  - "What do the Lab findings indicate?"
  - "Which guideline recommends this course of action?"
- RAG citations rendered inline with responses.

### 11.3 Key React Components
| Component | Description |
|-----------|-------------|
| `<CXRViewer />` | DICOM/PNG viewer with Grad-CAM toggle overlay |
| `<ECGChart />` | Recharts-based 12-lead waveform visualization |
| `<LabTable />` | Sortable lab table with abnormal-value highlighting |
| `<RiskDial />` | Circular gauge displaying risk percentage |
| `<ContributionChart />` | Horizontal stacked bar for modality attribution |
| `<MedicalChatbot />` | Full chat UI with streaming SSE support |
| `<RAGCitation />` | Inline citation card for referenced guidelines |

---

## 12. Explainable AI (XAI) & RAG Chatbot

### 12.1 Gradient-Weighted Class Activation Maps (Grad-CAM)
Applied to the ConvNeXt-Tiny Vision encoder to highlight which regions of the CXR drove the model's decision. The heatmap is superimposed on the original X-ray in the UI.

**Example Output:**  
For Heart Failure prediction: The Grad-CAM map highlights bilateral perihilar haziness and an enlarged cardiac silhouette — regions clinically consistent with pulmonary edema and cardiomegaly.

### 12.2 SHAP / Integrated Gradients for Modality Attribution
Applied to the intermediate concatenation layer of the Fusion MLP to compute feature attribution scores per modality embedding.
- **What it answers:** "Of the 1088 dimensions fed into the Fusion MLP, what fraction came from Vision, Signal, and Clinical that most influenced the output?"
- **Output:** A percentage triplet (e.g., Vision: 15%, Signal: 25%, Clinical: 60%).

### 12.3 RAG-Powered Medical LLM Chatbot

**Architecture:**
```
User Query
    ↓
Query Embedding (MiniLM)
    ↓
ChromaDB Similarity Search → Top-K Medical Guideline Chunks
    ↓
Prompt Construction:
    [System Context: Patient data, prediction scores, contributions]
    + [Retrieved Guidelines: "AHA 2022: BNP > 400 confirms HF..."]
    + [User Query]
    ↓
LLM Inference (Llama 3 / GPT-4o)
    ↓
Streamed Response + Inline Citations
```

**Example Interaction:**

*User:* "Why did the AI predict 82% Heart Failure probability for this patient?"

*VisionCare AI (streamed):* "The model's prediction was driven predominantly by the Clinical Laboratory data (60% contribution). The patient's BNP level of 850 pg/mL is severely elevated; according to the 2022 AHA/ACC Heart Failure Guidelines [Section 5.2, p.14], a BNP level exceeding 400 pg/mL in the absence of renal failure is a Class I indication for heart failure. This was corroborated by the ECG (25% contribution), which shows atrial fibrillation — a common comorbidity that worsens cardiac output. The Chest X-Ray (15% contribution) confirmed mild pulmonary vascular congestion. Taken together across all three modalities, the model's confidence of 82% aligns with established clinical diagnostic criteria."

---

## 13. Project Execution Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| **Phase 1** (Complete) | Data preprocessing, 8-model training, evaluation, visualization | 6 weeks | ✅ Done |
| **Phase 2 — Week 1** | MIMIC-IV label extraction (HF, Mortality), updated dataset creation | 1 week | 🔄 In Progress |
| **Phase 2 — Week 2** | Transfer learning fine-tuning of encoder heads + Fusion retraining | 1 week | ⬜ Pending |
| **Phase 2 — Week 3** | FastAPI backend scaffolding + PostgreSQL schema deployment | 1 week | ⬜ Pending |
| **Phase 2 — Week 4** | SHAP/Grad-CAM XAI integration + ChromaDB medical literature ingestion | 1 week | ⬜ Pending |
| **Phase 2 — Week 5** | React frontend development (Dashboard + Chatbot UI) | 1 week | ⬜ Pending |
| **Phase 2 — Week 6** | End-to-end integration testing, report writing, presentation prep | 1 week | ⬜ Pending |

---

## 14. Expected Outcomes & Success Metrics

| Metric | Phase 1 Result | Phase 2 Target | Rationale |
|--------|---------------|----------------|-----------|
| **Fusion AUC (vs. best single)** | +0.08% | **+13–17%** | Clinical outcome targets require all modalities |
| **Best Single-Modality AUC** | 0.7694 (Vision) | ~0.65–0.67 | Vision advantage is reduced for systemic diseases |
| **Fusion AUC (absolute)** | 0.7702 | **0.78–0.82** | Consistent with published HAIM benchmarks |
| **Model Retraining Time** | ∼4.6 hours (full) | **∼15–30 min** | Transfer learning reuses 95% of parameters |
| **System Latency (per inference)** | N/A | **< 2 seconds** | FastAPI async serving with pre-loaded models |
| **Chatbot Citation Accuracy** | N/A | **Grounded in evidence** | All responses backed by ChromaDB retrieval |

---

## 15. References

1. Holste, G., et al. (2024). *SYMILE-MIMIC: A Multi-modal Clinical Dataset.* PhysioNet.
2. Soenksen, L. R., et al. (2022). *Integrated Multimodal Artificial Intelligence Framework for Healthcare Applications.* Nature NPJ Digital Medicine.
3. Hayashi, T., et al. (2022). *MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images.* MLHC 2022.
4. Rajpurkar, P., et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays.* Stanford AI Lab.
5. Heidenreich, P.A., et al. (2022). *2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure.* Journal of the American College of Cardiology.
6. Johnson, A., et al. (2023). *MIMIC-IV (version 2.2).* PhysioNet.
7. Selvaraju, R.R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
8. Lundberg, S.M., et al. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS 2017.

---

*This document is intended as a living specification. Sections will be updated as implementation progresses.*
