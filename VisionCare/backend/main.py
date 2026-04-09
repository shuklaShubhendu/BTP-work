"""
VisionCare 2.0 — FastAPI Backend
=================================
MODEL LOADING:
  ┌─ FUSION MODELS (multi-modal) ────────────────────────────────────────────┐
  │  V2:  backend/models/fusion_v2_best.pth                 ← ✅ EXISTS     │
  │  V3:  backend/models/fusion_v3_best.pth                 ← download opt  │
  └───────────────────────────────────────────────────────────────────────────┘
  ┌─ SINGLE ENCODER MODELS ───────────────────────────────────────────────────┐
  │  CXR-only:  backend/models/cxr_only_best.pth                              │
  │             → Train with colab_train_vision.py (DenseNet121 / ConvNeXt)  │
  │  ECG-only:  backend/models/ecg_only_best.pth                              │
  │             → Train with colab_train_signal.py (1D-CNN / ResNet-1D)      │
  │  Labs-only: backend/models/labs_only_best.pth                             │
  │             → Train with colab_train_clinical.py (MLP, input_dim=100)    │
  │                                                                            │
  │  Place the .pth files in backend/models/ and restart the server.          │
  │  If missing → endpoint falls back to calibrated mock predictions.         │
  └───────────────────────────────────────────────────────────────────────────┘
  ┌─ CXR IMAGES ───────────────────────────────────────────────────────────────┐
  │  Place in: backend/demo_images/                                            │
  │  Naming:   p{patient_id}_e{encounter_id}.jpg                               │
  │  e.g.:     p1_e032.jpg, p3_e019.jpg                                       │
  │  Served at: http://127.0.0.1:8000/images/{filename}                        │
  └───────────────────────────────────────────────────────────────────────────┘
"""

import os, io, json, time, random, tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_VERSION  = "V2"           # Change to "V3" to switch
MODELS_DIR     = Path(__file__).parent / "models"
IMAGES_DIR     = Path(__file__).parent / "demo_images"
IMAGES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="VisionCare 2.0 API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Serve CXR images as static files
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# ─── MODEL LOADING ────────────────────────────────────────────────────────────
_model = None
_model_loaded = False

# Single-modal models
_cxr_model  = None
_ecg_model  = None
_labs_model = None

def try_load_model():
    """
    Attempt to load the trained VisionCare fusion model.
    Falls back to mock inference if model checkpoint not found.
    """
    global _model, _model_loaded
    checkpoint_path = MODELS_DIR / f"fusion_{MODEL_VERSION.lower()}_best.pth"
    
    if not checkpoint_path.exists():
        print(f"⚠️  No checkpoint at {checkpoint_path} — running in MOCK mode")
        print(f"    To enable real inference, place your checkpoint at:")
        print(f"    {checkpoint_path.absolute()}")
        return False
    
    try:
        import torch
        print(f"📦 Loading {MODEL_VERSION} checkpoint from {checkpoint_path}...")
        
        # Import your actual model architecture
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        if MODEL_VERSION == "V2":
            # V2 uses colab_fusion_v2.py architecture
            from colab_fusion_v2 import VisionCare as Model
        else:
            # V3 uses colab_fusion_v3.py architecture
            from colab_fusion_v3 import VisionCare as Model
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle EMA weights (the best checkpoint saves EMA state)
        state_dict = checkpoint.get("ema_state_dict",
                     checkpoint.get("model_state_dict",
                     checkpoint.get("state_dict", checkpoint)))
        
        _model = Model().to(device)
        _model.load_state_dict(state_dict, strict=False)
        _model.eval()
        _model_loaded = True
        print(f"✅ {MODEL_VERSION} model loaded — device: {device}")
        return True
        
    except ImportError as e:
        print(f"⚠️  PyTorch not available in backend env: {e}")
        print("    Running in MOCK mode. Install torch to enable real inference.")
        return False
    except Exception as e:
        print(f"⚠️  Model load failed: {e}")
        return False

# Try loading on startup
try_load_model()

def _load_single(ckpt_path, architectures, model_kwargs=None):
    """
    Generic single-model loader.
    Tries each architecture class in order; returns the first that loads successfully.
    """
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt.get("state_dict", ckpt)))
        saved_name = ckpt.get("model_name", "")
        kwargs = model_kwargs or {}

        # Prefer the architecture whose name matches what was saved
        ordered = sorted(architectures, key=lambda C: 0 if getattr(C, 'name', '') == saved_name else 1)
        for Cls in ordered:
            try:
                m = Cls(**kwargs)
                m.load_state_dict(state, strict=False)
                m.to(device).eval()
                print(f"✅ Loaded {Cls.name} from {ckpt_path.name}")
                return m
            except Exception:
                continue
        print(f"⚠️  No architecture matched for {ckpt_path.name}")
        return None
    except Exception as e:
        print(f"⚠️  Failed to load {ckpt_path}: {e}")
        return None

def try_load_single_models():
    global _cxr_model, _ecg_model, _labs_model
    try:
        from architectures import CXR_ARCHITECTURES, ECG_ARCHITECTURES, LABS_ARCHITECTURES
    except ImportError as e:
        print(f"⚠️  architectures.py not importable: {e}")
        return

    cxr_path  = MODELS_DIR / "cxr_only_best.pth"
    ecg_path  = MODELS_DIR / "ecg_only_best.pth"
    labs_path = MODELS_DIR / "labs_only_best.pth"

    if cxr_path.exists():
        _cxr_model  = _load_single(cxr_path,  CXR_ARCHITECTURES)
    else:
        print(f"ℹ️  CXR model not found  → {cxr_path.name} (mock will be used)")

    if ecg_path.exists():
        _ecg_model  = _load_single(ecg_path,  ECG_ARCHITECTURES)
    else:
        print(f"ℹ️  ECG model not found  → {ecg_path.name} (mock will be used)")

    if labs_path.exists():
        _labs_model = _load_single(labs_path, LABS_ARCHITECTURES, {"input_dim": 100})
    else:
        print(f"ℹ️  Labs model not found → {labs_path.name} (mock will be used)")

try_load_single_models()

def run_inference(cxr_path=None, ecg_array=None, labs_dict=None):
    """Run real or mock inference."""
    if not _model_loaded:
        return mock_inference(labs_dict)
    
    try:
        import torch
        import numpy as np
        from PIL import Image
        import torchvision.transforms as T
        
        device = next(_model.parameters()).device
        
        # Preprocess CXR
        if cxr_path and Path(cxr_path).exists():
            img = Image.open(cxr_path).convert("RGB")
            transform = T.Compose([T.Resize((320, 320)), T.ToTensor(),
                                   T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            cxr_tensor = transform(img).unsqueeze(0).to(device)
        else:
            cxr_tensor = torch.zeros(1, 3, 320, 320).to(device)
        
        # Preprocess Labs
        LAB_MEANS =  [0.15, 0.02, 1.0, 139, 4.0, 13.0, 7.5, 100, 1.5, 3.5]
        LAB_STDS  =  [0.10, 0.02, 0.4,   5, 0.6,  2.5, 2.5,  40, 0.5, 1.0]
        lab_vals = [
            float(labs_dict.get("bnp",       0.15)),
            float(labs_dict.get("troponin",  0.02)),
            float(labs_dict.get("creatinine",1.0 )),
            float(labs_dict.get("sodium",    139  )),
            float(labs_dict.get("potassium", 4.0  )),
            float(labs_dict.get("hemoglobin",13.0 )),
            float(labs_dict.get("wbc",       7.5  )),
            float(labs_dict.get("glucose",   100  )),
            float(labs_dict.get("bun",       15   )),
            float(labs_dict.get("albumin",   3.5  )),
        ]
        lab_norm = [(v - m) / s for v, m, s in zip(lab_vals, LAB_MEANS, LAB_STDS)]
        lab_tensor = torch.tensor(lab_norm, dtype=torch.float32).unsqueeze(0).to(device)
        
        # ECG
        if ecg_array is not None:
            ecg_tensor = torch.tensor(ecg_array, dtype=torch.float32).to(device)
            if ecg_tensor.dim() == 1: ecg_tensor = ecg_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            ecg_tensor = torch.zeros(1, 1, 5000, 12).to(device)
        
        with torch.no_grad():
            out = _model(cxr_tensor, ecg_tensor, lab_tensor)
            logits = out.get("logits", out.get("out", None))
            gates  = out.get("gates", {"vision": 0.33, "signal": 0.33, "clinical": 0.34})
            
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            mort_prob = float(probs[0]) * 100
            hf_prob   = float(probs[1]) * 100
            
            return {
                "hf_risk":        round(hf_prob, 1),
                "mortality_risk": round(mort_prob, 1),
                "gates": {
                    "vision":   float(gates.get("vision",   0.33)),
                    "signal":   float(gates.get("signal",   0.33)),
                    "clinical": float(gates.get("clinical", 0.34)),
                },
                "model_version": MODEL_VERSION,
                "real_inference": True,
            }
    except Exception as e:
        print(f"Inference error: {e}")
        return mock_inference(labs_dict)

def mock_inference(labs_dict=None):
    """Calibrated mock predictions based on lab values."""
    labs = labs_dict or {}
    score = 0
    if float(labs.get("bnp",        0)) > 400:  score += 35
    if float(labs.get("troponin",   0)) > 0.04: score += 20
    if float(labs.get("creatinine", 0)) > 1.5:  score += 15
    if float(labs.get("sodium",     140)) < 135: score += 12
    if float(labs.get("hemoglobin", 13)) < 12:   score += 8
    if float(labs.get("glucose",    90)) > 120:  score += 5
    hf_risk = min(score + random.randint(5, 15), 95)
    mort_risk = min(round(hf_risk * 0.38 + random.randint(0, 5)), 90)
    return {
        "hf_risk": hf_risk, "mortality_risk": mort_risk,
        "gates": {"vision": 0.33, "signal": 0.33, "clinical": 0.34},
        "model_version": MODEL_VERSION, "real_inference": False,
    }

# ─── PATIENT DATABASE (expand as needed) ─────────────────────────────────────
PATIENTS = {
    "1": {"id":"1","mrn":"MRN-004821","name":"Rajesh Kumar",  "age":67,"gender":"Male",  "status":"Active",    "condition":"Acute decompensated HF",     "severity":"critical"},
    "2": {"id":"2","mrn":"MRN-003156","name":"Priya Sharma",  "age":54,"gender":"Female","status":"Active",    "condition":"Follow-up cardiac evaluation","severity":"moderate"},
    "3": {"id":"3","mrn":"MRN-005934","name":"Anand Patel",   "age":72,"gender":"Male",  "status":"Active",    "condition":"Severe HF with reduced EF",   "severity":"critical"},
    "4": {"id":"4","mrn":"MRN-002187","name":"Meera Iyer",    "age":49,"gender":"Female","status":"Discharged","condition":"Routine cardiac screening",   "severity":"normal"},
    "5": {"id":"5","mrn":"MRN-006743","name":"Vikram Singh",  "age":61,"gender":"Male",  "status":"Active",    "condition":"Moderate HF follow-up",       "severity":"moderate"},
    "6": {"id":"6","mrn":"MRN-001298","name":"Lakshmi Nair",  "age":58,"gender":"Female","status":"Discharged","condition":"Post-MI cardiac evaluation", "severity":"normal"},
    "7": {"id":"7","mrn":"MRN-007512","name":"Suresh Reddy",  "age":75,"gender":"Male",  "status":"Active",    "condition":"Chronic HF management",       "severity":"critical"},
    "8": {"id":"8","mrn":"MRN-008901","name":"Deepa Menon",   "age":45,"gender":"Female","status":"Active",    "condition":"New onset dyspnoea workup",   "severity":"moderate"},
}

ENCOUNTERS = {
"1": [
  {"id":"e032","label":"E-032","date":"26 Mar 2026","description":"Acute decompensated HF",
   "hf_risk":82,"mortality_risk":31,
   "gates":{"vision":0.15,"signal":0.25,"clinical":0.60},
   "cxr_findings":["Cardiomegaly","Pulmonary oedema","Bilateral pleural effusion"],
   "ecg_findings":["Atrial fibrillation","Left ventricular hypertrophy","ST changes"],
   "cxr_image":"p1_e032.jpg",
   "labs":[
     {"name":"BNP","value":"850 pg/mL","normal":"<100","status":"Critical"},
     {"name":"Troponin I","value":"9.45 ng/mL","normal":"<0.04","status":"Critical"},
     {"name":"Creatinine","value":"1.8 mg/dL","normal":"0.7-1.3","status":"High"},
     {"name":"Sodium","value":"138 mEq/L","normal":"136-145","status":"Normal"},
     {"name":"Potassium","value":"4.2 mEq/L","normal":"3.5-5.0","status":"Normal"},
     {"name":"Hemoglobin","value":"11.2 g/dL","normal":"13.5-17.5","status":"Low"},
     {"name":"WBC","value":"9.8 ×10³/μL","normal":"4.5-11.0","status":"Normal"},
     {"name":"Glucose","value":"142 mg/dL","normal":"70-100","status":"High"},
   ]},
  {"id":"e028","label":"E-028","date":"14 Feb 2026","description":"Follow-up cardiac evaluation",
   "hf_risk":58,"mortality_risk":19,
   "gates":{"vision":0.45,"signal":0.30,"clinical":0.25},
   "cxr_findings":["Mild cardiomegaly","Mild pulmonary congestion"],
   "ecg_findings":["Atrial fibrillation","Controlled ventricular rate"],
   "cxr_image":"p1_e028.jpg",
   "labs":[
     {"name":"BNP","value":"420 pg/mL","normal":"<100","status":"High"},
     {"name":"Creatinine","value":"1.4 mg/dL","normal":"0.7-1.3","status":"High"},
   ]},
],
"2": [
  {"id":"e041","label":"E-041","date":"24 Mar 2026","description":"Follow-up cardiac evaluation",
   "hf_risk":45,"mortality_risk":18,"gates":{"vision":0.50,"signal":0.20,"clinical":0.30},
   "cxr_findings":["Mild cardiomegaly"],"ecg_findings":["Normal sinus rhythm","Left axis deviation"],
   "cxr_image":"p2_e041.jpg",
   "labs":[{"name":"BNP","value":"290 pg/mL","normal":"<100","status":"High"}]},
],
"3": [
  {"id":"e019","label":"E-019","date":"22 Mar 2026","description":"Severe HF with reduced EF",
   "hf_risk":91,"mortality_risk":56,"gates":{"vision":0.20,"signal":0.30,"clinical":0.50},
   "cxr_findings":["Severe cardiomegaly","Pulmonary oedema","Pleural effusion"],
   "ecg_findings":["Left bundle branch block","ST depression"],
   "cxr_image":"p3_e019.jpg",
   "labs":[
     {"name":"BNP","value":"2100 pg/mL","normal":"<100","status":"Critical"},
     {"name":"Creatinine","value":"2.4 mg/dL","normal":"0.7-1.3","status":"Critical"},
     {"name":"Sodium","value":"128 mEq/L","normal":"136-145","status":"Critical"},
   ]},
],
"4": [{"id":"e004","label":"E-004","date":"20 Mar 2026","description":"Routine cardiac screening","hf_risk":12,"mortality_risk":5,"gates":{"vision":0.65,"signal":0.20,"clinical":0.15},"cxr_findings":["No significant findings"],"ecg_findings":["Normal sinus rhythm"],"cxr_image":"p4_e004.jpg","labs":[{"name":"BNP","value":"45 pg/mL","normal":"<100","status":"Normal"}]}],
"5": [{"id":"e005","label":"E-005","date":"18 Mar 2026","description":"Moderate HF follow-up","hf_risk":67,"mortality_risk":28,"gates":{"vision":0.35,"signal":0.30,"clinical":0.35},"cxr_findings":["Cardiomegaly","Mild pulmonary congestion"],"ecg_findings":["Atrial fibrillation","Rate-controlled"],"cxr_image":"p5_e005.jpg","labs":[{"name":"BNP","value":"580 pg/mL","normal":"<100","status":"Critical"}]}],
"6": [{"id":"e006","label":"E-006","date":"15 Mar 2026","description":"Post-MI cardiac evaluation","hf_risk":23,"mortality_risk":8,"gates":{"vision":0.55,"signal":0.25,"clinical":0.20},"cxr_findings":["Mild atelectasis"],"ecg_findings":["Q waves in V1-V3"],"cxr_image":"p6_e006.jpg","labs":[{"name":"BNP","value":"120 pg/mL","normal":"<100","status":"High"}]}],
"7": [{"id":"e007","label":"E-007","date":"12 Mar 2026","description":"Chronic HF management","hf_risk":78,"mortality_risk":42,"gates":{"vision":0.25,"signal":0.35,"clinical":0.40},"cxr_findings":["Cardiomegaly","Pleural effusion"],"ecg_findings":["Ventricular tachycardia","LV hypertrophy"],"cxr_image":"p7_e007.jpg","labs":[{"name":"BNP","value":"1200 pg/mL","normal":"<100","status":"Critical"}]}],
"8": [{"id":"e008","label":"E-008","date":"10 Mar 2026","description":"New onset dyspnea workup","hf_risk":34,"mortality_risk":12,"gates":{"vision":0.45,"signal":0.25,"clinical":0.30},"cxr_findings":["Borderline heart size"],"ecg_findings":["Normal sinus rhythm"],"cxr_image":"p8_e008.jpg","labs":[{"name":"BNP","value":"220 pg/mL","normal":"<100","status":"High"}]}],
}

# ─── HELPER ───────────────────────────────────────────────────────────────────
def resolve_image(filename: str) -> Optional[str]:
    """Return URL if image file exists, else None."""
    if filename and (IMAGES_DIR / filename).exists():
        return f"http://127.0.0.1:8000/images/{filename}"
    return None

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "VisionCare 2.0", "model": MODEL_VERSION,
            "model_loaded": _model_loaded, "images_dir": str(IMAGES_DIR)}

@app.get("/api/stats")
def stats():
    return {"total_patients": 10842, "high_risk_hf": 2341,
            "critical_cases": 156, "predictions_today": 89,
            "model_version": MODEL_VERSION, "model_loaded": _model_loaded}

@app.get("/api/patients")
def get_patients(search: str = ""):
    pts = list(PATIENTS.values())
    if search:
        pts = [p for p in pts if search.lower() in p["name"].lower()
               or search.lower() in p["mrn"].lower()]
    # Attach latest encounter risk scores
    for p in pts:
        encs = ENCOUNTERS.get(p["id"], [])
        if encs:
            p["hf_risk"]         = encs[0]["hf_risk"]
            p["mortality_risk"]  = encs[0]["mortality_risk"]
            p["last_encounter"]  = encs[0]["date"]
            p["cxr_image"]       = resolve_image(encs[0].get("cxr_image",""))
    return pts

@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str):
    p = PATIENTS.get(patient_id)
    if not p: raise HTTPException(404, "Patient not found")
    encs = ENCOUNTERS.get(patient_id, [])
    if encs:
        p["hf_risk"]        = encs[0]["hf_risk"]
        p["mortality_risk"] = encs[0]["mortality_risk"]
        p["last_encounter"] = encs[0]["date"]
    return p

@app.get("/api/patients/{patient_id}/encounters")
def get_encounters(patient_id: str):
    encs = ENCOUNTERS.get(patient_id, [])
    for e in encs:
        e["cxr_image_url"] = resolve_image(e.get("cxr_image", ""))
    return encs

@app.get("/api/patients/{patient_id}/encounters/{encounter_id}")
def get_encounter(patient_id: str, encounter_id: str):
    encs = ENCOUNTERS.get(patient_id, [])
    enc = next((e for e in encs if e["id"] == encounter_id), None)
    if not enc: raise HTTPException(404, "Encounter not found")
    enc = dict(enc)
    enc["cxr_image_url"] = resolve_image(enc.get("cxr_image", ""))
    enc["patient"] = PATIENTS.get(patient_id)
    return enc

@app.get("/api/model/info")
def model_info():
    v2 = {"macro_auc": 0.8105, "hf_auc": 0.8189, "mort_auc": 0.8022,
           "encoders": {"cxr":"ConvNeXt-Tiny","ecg":"1D-CNN","labs":"2-layer MLP"},
           "checkpoint": str(MODELS_DIR / "fusion_v2_best.pth"),
           "exists": (MODELS_DIR / "fusion_v2_best.pth").exists()}
    v3 = {"macro_auc": 0.7904, "hf_auc": 0.7519, "mort_auc": 0.8289,
           "encoders": {"cxr":"ResNet-50/SYMILE","ecg":"1D-ResNet18","labs":"3-layer NN"},
           "checkpoint": str(MODELS_DIR / "fusion_v3_best.pth"),
           "exists": (MODELS_DIR / "fusion_v3_best.pth").exists()}
    return {"active_version": MODEL_VERSION, "v2": v2, "v3": v3,
            "model_loaded": _model_loaded}

# ─── SINGLE-MODAL INFERENCE HELPERS ──────────────────────────────────────────

def _run_cxr_model(image_bytes: bytes) -> dict:
    """Run CXR-only model on raw image bytes. Falls back to mock on failure."""
    if _cxr_model is None:
        return mock_inference()

    try:
        import torch
        from PIL import Image
        import torchvision.transforms as T
        from architectures import map_6labels_to_risks

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        x = transform(img).unsqueeze(0).to(next(_cxr_model.parameters()).device)
        with torch.no_grad():
            logits, _ = _cxr_model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0].tolist()

        risks = map_6labels_to_risks(probs)
        return {
            **risks,
            "gates": {"vision": 1.0, "signal": 0.0, "clinical": 0.0},
            "model_version": f"CXR-{getattr(_cxr_model, 'name', 'single')}",
            "real_inference": True,
        }
    except Exception as e:
        print(f"CXR inference error: {e}")
        return mock_inference()


def _run_ecg_model(ecg_array=None) -> dict:
    """
    Run ECG-only model.
    ecg_array: numpy array shape (12, seq_len) or None → uses synthetic demo signal.
    """
    if _ecg_model is None:
        return mock_inference()

    try:
        import torch
        import numpy as np
        from architectures import map_6labels_to_risks

        device = next(_ecg_model.parameters()).device
        if ecg_array is not None:
            arr = np.array(ecg_array, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(12, -1)
        else:
            # Synthetic normal sinus rhythm demo — 12 leads × 5000 samples
            t = np.linspace(0, 10, 5000)
            base = np.sin(2 * np.pi * 1.2 * t)          # ~72 bpm
            arr  = np.stack([base + 0.05 * np.random.randn(5000) for _ in range(12)])
            arr  = arr.astype(np.float32)

        x = torch.tensor(arr).unsqueeze(0).to(device)   # (1, 12, seq_len)
        with torch.no_grad():
            logits, _ = _ecg_model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0].tolist()

        risks = map_6labels_to_risks(probs)
        return {
            **risks,
            "gates": {"vision": 0.0, "signal": 1.0, "clinical": 0.0},
            "model_version": f"ECG-{getattr(_ecg_model, 'name', 'single')}",
            "real_inference": True,
        }
    except Exception as e:
        print(f"ECG inference error: {e}")
        return mock_inference()


def _run_labs_model(labs_dict: dict) -> dict:
    """
    Run Labs-only model.
    Constructs a 100-dim input from the 8 spot-lab values:
      dims 0-7  : normalised lab values (BNP, troponin, creatinine, Na, K, Hb, WBC, glucose)
      dims 8-15 : missingness flags (1 = value was provided)
      dims 16-99: zero-padded (remaining MIMIC percentile features not available here)
    """
    if _labs_model is None:
        return mock_inference(labs_dict)

    try:
        import torch
        import numpy as np
        from architectures import map_6labels_to_risks

        device = next(_labs_model.parameters()).device

        LAB_KEYS  = ["bnp", "troponin", "creatinine", "sodium", "potassium", "hemoglobin", "wbc", "glucose"]
        LAB_MEANS = [0.15,   0.02,       1.0,          139,      4.0,         13.0,          7.5,  100]
        LAB_STDS  = [0.10,   0.02,       0.4,            5,      0.6,          2.5,           2.5,   40]

        vals  = [float(labs_dict.get(k, m)) for k, m in zip(LAB_KEYS, LAB_MEANS)]
        norms = [(v - m) / s for v, m, s in zip(vals, LAB_MEANS, LAB_STDS)]
        miss  = [1.0 if labs_dict.get(k) not in (None, "", 0) else 0.0 for k in LAB_KEYS]

        feat = np.zeros(100, dtype=np.float32)
        feat[0:8]  = norms
        feat[8:16] = miss
        # feat[16:100] = 0  (unknown percentile features)

        x = torch.tensor(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = _labs_model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0].tolist()

        risks = map_6labels_to_risks(probs)
        return {
            **risks,
            "gates": {"vision": 0.0, "signal": 0.0, "clinical": 1.0},
            "model_version": f"Labs-{getattr(_labs_model, 'name', 'single')}",
            "real_inference": True,
        }
    except Exception as e:
        print(f"Labs inference error: {e}")
        return mock_inference(labs_dict)


class AnalyzeRequest(BaseModel):
    labs: dict = {}
    has_ecg: bool = False
    mode: str = "multimodal"   # multimodal | cxr | ecg | labs

class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[str] = None
    encounter_id: Optional[str] = None

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    start = time.time()
    result = run_inference(labs_dict=req.labs)
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    return result


@app.post("/api/analyze/cxr")
async def analyze_cxr(file: UploadFile = File(...)):
    """CXR-only single-modal inference. Accepts any image file (JPEG/PNG)."""
    start = time.time()
    image_bytes = await file.read()
    result = _run_cxr_model(image_bytes)
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    result["mode"] = "cxr"
    return result


@app.post("/api/analyze/ecg")
def analyze_ecg():
    """ECG-only single-modal inference. Uses demo synthetic signal when no file provided."""
    start = time.time()
    result = _run_ecg_model(ecg_array=None)  # demo signal; extend later for file upload
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    result["mode"] = "ecg"
    return result


@app.post("/api/analyze/labs")
def analyze_labs(req: AnalyzeRequest):
    """Labs-only single-modal inference."""
    start = time.time()
    result = _run_labs_model(req.labs)
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    result["mode"] = "labs"
    return result


@app.get("/api/models/status")
def models_status():
    """Returns which models are loaded."""
    return {
        "fusion":    {"loaded": _model_loaded,   "version": MODEL_VERSION},
        "cxr_only":  {"loaded": _cxr_model  is not None, "name": getattr(_cxr_model,  'name', None)},
        "ecg_only":  {"loaded": _ecg_model  is not None, "name": getattr(_ecg_model,  'name', None)},
        "labs_only": {"loaded": _labs_model is not None, "name": getattr(_labs_model, 'name', None)},
    }

@app.post("/api/chat")
def chat(req: ChatRequest):
    """Simple rule-based fallback. Gemini is called client-side from React."""
    return {"reply": f"Backend received: {req.message[:60]}. Use Gemini API key for full AI chat.",
            "source": "backend_fallback"}
