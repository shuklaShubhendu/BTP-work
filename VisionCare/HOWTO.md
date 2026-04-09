# VisionCare 2.0 — Complete Setup Guide

## ✅ Step 1: Start Backend (without Docker)

```powershell
cd "btp sem 7\VisionCare\backend"
pip install fastapi uvicorn pillow numpy
python -m uvicorn main:app --reload --port 8000
```
Backend runs at **http://127.0.0.1:8000**

---

## ✅ Step 2: Start Frontend (without Docker)

```powershell
cd "btp sem 7\VisionCare\visioncare-app"
npm run dev
```
Frontend runs at **http://localhost:5173**

---

## 🐳 Docker (one command for both)

```powershell
cd "btp sem 7\VisionCare"

# Build and start both services
docker-compose up --build

# With Gemini key:
$env:VITE_GEMINI_KEY="AIzaSy_your_key_here"
docker-compose up --build
```

| Service  | URL                    |
|----------|------------------------|
| Frontend | http://localhost:3000  |
| Backend  | http://localhost:8000  |

---

## 🤖 Gemini API Key

1. Go to https://aistudio.google.com/app/apikey → Create API Key (free)
2. Open `visioncare-app/.env` and set:
   ```
   VITE_GEMINI_KEY=AIzaSy_your_key_here
   ```
3. Restart `npm run dev`

---

## 📥 What to Download from Google Drive

### From `/symile-mimic/VisionCare_V2/` folder:

| File on Drive | Save as (locally) | Purpose |
|---------------|-------------------|---------|
| `best_model.pth` | `backend/models/fusion_v2_best.pth` | **V2 model checkpoint (REQUIRED for real inference)** |
| `VisionCare_V2/best_model.pth` is usually named `checkpoint_best.pth` or `ema_best.pth` | same | — |

### From `/symile-mimic/VisionCare_V3/` folder (optional):

| File on Drive | Save as (locally) |
|---------------|-------------------|
| `best_model.pth` | `backend/models/fusion_v3_best.pth` |

> Without these files, backend runs in **MOCK mode** — fully functional for the UI demo

---

## 🫁 CXR Demo Images — Naming Convention

### Image file naming: `p{patient_id}_e{encounter_id}.jpg`

| Patient | Severity | Encounter | Image filename | What to show |
|---------|----------|-----------|----------------|--------------|
| Rajesh Kumar (ID=1) | 🔴 Critical | E-032 | `p1_e032.jpg` | Cardiomegaly + Pulmonary Oedema |
| Rajesh Kumar (ID=1) | 🔴 Critical | E-028 | `p1_e028.jpg` | Mild cardiomegaly |
| Priya Sharma (ID=2) | 🟡 Moderate | E-041 | `p2_e041.jpg` | Mild cardiomegaly |
| Anand Patel (ID=3)  | 🔴 Critical | E-019 | `p3_e019.jpg` | Severe HF + Pleural effusion |
| Meera Iyer (ID=4)   | 🟢 Normal   | E-004 | `p4_e004.jpg` | Normal CXR |
| Vikram Singh (ID=5) | 🟡 Moderate | E-005 | `p5_e005.jpg` | Mild cardiomegaly |
| Lakshmi Nair (ID=6) | 🟢 Normal   | E-006 | `p6_e006.jpg` | Mild atelectasis |
| Suresh Reddy (ID=7) | 🔴 Critical | E-007 | `p7_e007.jpg` | Cardiomegaly + Effusion |
| Deepa Menon (ID=8)  | 🟡 Moderate | E-008 | `p8_e008.jpg` | Borderline heart size |

### Where to place them: `backend/demo_images/`

```
VisionCare/
└── backend/
    └── demo_images/       ← DROP FILES HERE
        ├── p1_e032.jpg    ← critical HF with oedema
        ├── p1_e028.jpg
        ├── p2_e041.jpg
        ├── p3_e019.jpg    ← most critical (91% HF risk)
        ├── p4_e004.jpg    ← normal CXR
        ├── p5_e005.jpg
        ├── p6_e006.jpg
        ├── p7_e007.jpg
        └── p8_e008.jpg
```

### Free image sources (no login needed):

**NIH ChestX-ray14:**
- Go to: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Download `images_001.tar.gz` → extract 10 images
- Rename to match the table above: cardiomegaly images → `p1_e032.jpg`, normal → `p4_e004.jpg`

**From MIMIC-CXR (you already have access):**
- PhysioNet: physionet.org/content/mimic-cxr
- Download any `.jpg` files from `files/p10/p10000032/` etc.
- Rename following the table above

---

## 🏗️ Does the Backend Run Your Trained Model?

**YES** — but requires:
1. `backend/models/fusion_v2_best.pth` to exist (downloaded from Google Drive)
2. PyTorch installed: `pip install torch torchvision`
3. The `colab_fusion_v2.py` file to be in the parent VisionCare folder (it already is ✅)

When model IS loaded: real softmax probabilities from VisionCare V2/V3 fusion
When model is NOT found: calibrated mock predictions (still clinically plausible)

The backend auto-detects the checkpoint on startup and prints:
- `✅ V2 model loaded — device: cuda/cpu`  ← real inference
- `⚠️ No checkpoint at ... — running in MOCK mode` ← mock inference

---

## 📁 Final Folder Structure

```
VisionCare/
├── visioncare-app/          ← React frontend
│   ├── Dockerfile
│   └── src/...
├── backend/
│   ├── main.py             ← FastAPI (loads real model if .pth exists)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── models/
│   │   ├── fusion_v2_best.pth   ← download from Google Drive
│   │   └── fusion_v3_best.pth   ← download from Google Drive (optional)
│   └── demo_images/
│       ├── p1_e032.jpg          ← CXR images you add
│       └── ...
├── docker-compose.yml       ← runs everything with one command
├── colab_fusion_v2.py       ← model architecture (used by backend)
├── colab_fusion_v3.py       ← V3 architecture (used when MODEL_VERSION=V3)
└── HOWTO.md
```
