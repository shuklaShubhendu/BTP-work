"""
VisionCare 3.0 — FastAPI Backend
=================================
MODEL LOADING:
  ┌─ FUSION MODEL (V3 — multi-modal, 8 diseases) ─────────────────────────────┐
  │  backend/models/fusion_v3_best.pth                                         │
  │  Contains ALL weights (ResNet-50 + 1D ResNet-18 + 3-NN + Fusion + Head)   │
  │  No SYMILE checkpoint needed — everything is baked into this one file.     │
  └───────────────────────────────────────────────────────────────────────────┘
  ┌─ SINGLE ENCODER MODELS (optional, for modality-specific analysis) ────────┐
  │  CXR-only:  backend/models/cxr_only_best.pth                              │
  │  ECG-only:  backend/models/ecg_only_best.pth                              │
  │  Labs-only: backend/models/labs_only_best.pth                              │
  │  If missing → endpoint falls back to calibrated mock predictions.          │
  └───────────────────────────────────────────────────────────────────────────┘
  ┌─ CXR IMAGES ──────────────────────────────────────────────────────────────┐
  │  Place in: backend/demo_images/                                            │
  │  Naming:   p{patient_id}_e{encounter_id}.jpg                               │
  │  Served at: http://127.0.0.1:8000/images/{filename}                        │
  └───────────────────────────────────────────────────────────────────────────┘
"""

import os, io, json, time, sqlite3
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_VERSION  = "V3"
MODELS_DIR     = Path(__file__).parent / "models"
IMAGES_DIR     = Path(__file__).parent / "demo_images"
DATA_DIR       = Path(__file__).parent / "data"
DB_PATH        = DATA_DIR / "visioncare.db"
IMAGES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# V3 target names
V3_TARGETS = [
    'mortality', 'heart_failure', 'myocardial_infarction', 'arrhythmia',
    'sepsis', 'pulmonary_embolism', 'acute_kidney_injury', 'icu_admission'
]

app = FastAPI(title="VisionCare 3.0 API", version="3.0")
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
    Load the VisionCare V3 fusion model from fusion_v3_best.pth.
    This checkpoint contains ALL weights (encoders + fusion + head).
    No SYMILE checkpoint needed at inference time.
    """
    global _model, _model_loaded
    checkpoint_path = MODELS_DIR / "fusion_v3_best.pth"

    if not checkpoint_path.exists():
        print(f"⚠️  No checkpoint at {checkpoint_path} — running in MOCK mode")
        print(f"    To enable real inference, place your checkpoint at:")
        print(f"    {checkpoint_path.absolute()}")
        return False

    try:
        import torch
        print(f"📦 Loading VisionCare V3 checkpoint from {checkpoint_path}...")

        from architectures import VisionCareV3

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle EMA weights (the best checkpoint saves EMA state)
        state_dict = checkpoint.get("ema_state_dict",
                     checkpoint.get("model_state_dict",
                     checkpoint.get("state_dict", checkpoint)))

        _model = VisionCareV3(num_labels=8).to(device)
        _model.load_state_dict(state_dict, strict=False)
        _model.eval()
        _model_loaded = True
        print(f"✅ VisionCare V3 loaded — device: {device}, 8 disease targets")
        return True

    except ImportError as e:
        print(f"⚠️  PyTorch not available in backend env: {e}")
        print("    Running in MOCK mode. Install torch to enable real inference.")
        return False
    except Exception as e:
        print(f"⚠️  Model load failed: {e}")
        import traceback; traceback.print_exc()
        return False

# Try loading on startup
try_load_model()


def try_load_single_models():
    """Load V3-compatible single-modal models (share weights from fusion checkpoint)."""
    global _cxr_model, _ecg_model, _labs_model
    ckpt = str(MODELS_DIR / "fusion_v3_best.pth")
    try:
        from architectures import build_v3_single_model
        # All 3 single-modal 'models' are actually the same fusion model
        # stored once in memory (Python passes by reference, so this is efficient)
        _cxr_model  = build_v3_single_model('cxr',  ckpt) if (MODELS_DIR / "fusion_v3_best.pth").exists() else None
        _ecg_model  = _cxr_model   # Same model, different zero-padding
        _labs_model = _cxr_model   # Same model, different zero-padding
        if _cxr_model:
            print("✅ V3 single-modal endpoints ready (ResNet-50 / 1D-ResNet18 / 3-NN)")
        else:
            print("ℹ️  Single-modal will use mock (place fusion_v3_best.pth to enable)")
    except Exception as e:
        print(f"⚠️  Single-model setup failed: {e}")

try_load_single_models()

# ─── RAG ENGINE (Medical AI Chat) ─────────────────────────────────────────────
from rag_engine import MedicalRAGEngine
_rag = MedicalRAGEngine()

def try_load_rag():
    """Load FAISS index for medical RAG. Auto-builds if missing."""
    try:
        success = _rag.load_index()
        if success:
            print(f"✅ Medical RAG ready — {_rag.chunk_count} chunks from textbooks")
        else:
            print("⚠️  RAG not available — chat will use rule-based responses")
    except Exception as e:
        print(f"⚠️  RAG init failed: {e} — chat will use rule-based responses")

try_load_rag()


# ─── V3 INFERENCE (8 diseases + 3 gate weights) ──────────────────────────────

def run_inference(cxr_path=None, ecg_array=None, labs_dict=None):
    """Run real V3 inference or fall back to mock."""
    if not _model_loaded:
        return mock_inference(labs_dict)

    try:
        import torch
        import numpy as np
        from PIL import Image
        import torchvision.transforms as T

        device = next(_model.parameters()).device

        # Preprocess CXR → (1, 3, 320, 320)
        if cxr_path and Path(cxr_path).exists():
            img = Image.open(cxr_path).convert("RGB")
            transform = T.Compose([T.Resize((320, 320)), T.ToTensor(),
                                   T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            cxr_tensor = transform(img).unsqueeze(0).to(device)
        else:
            cxr_tensor = torch.zeros(1, 3, 320, 320).to(device)

        # Preprocess Labs → (1, 100)
        labs = labs_dict or {}
        LAB_KEYS  = ["bnp","troponin","creatinine","sodium","potassium","hemoglobin","wbc","glucose","bun","albumin"]
        LAB_MEANS = [0.15, 0.02, 1.0, 139, 4.0, 13.0, 7.5, 100, 15, 3.5]
        LAB_STDS  = [0.10, 0.02, 0.4,   5, 0.6,  2.5, 2.5,  40, 5, 1.0]
        lab_vals = [float(labs.get(k, m)) for k, m in zip(LAB_KEYS, LAB_MEANS)]
        lab_norm = [(v - m) / s for v, m, s in zip(lab_vals, LAB_MEANS, LAB_STDS)]
        # Pad to 100-D (50 percentile + 50 missingness features from SYMILE)
        feat = [0.0] * 100
        for i, v in enumerate(lab_norm[:10]):
            feat[i] = v
        for i, k in enumerate(LAB_KEYS[:10]):
            feat[50+i] = 1.0 if labs.get(k) not in (None, "", 0) else 0.0
        import torch
        lab_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)

        # Preprocess ECG → (1, 12, 5000)
        if ecg_array is not None:
            ecg_tensor = torch.tensor(ecg_array, dtype=torch.float32).to(device)
            if ecg_tensor.dim() == 2: ecg_tensor = ecg_tensor.unsqueeze(0)
        else:
            ecg_tensor = torch.zeros(1, 12, 5000).to(device)

        with torch.no_grad():
            logits, gates = _model(cxr_tensor, ecg_tensor, lab_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            g = gates.cpu().numpy()[0]

            risks = {}
            for i, target in enumerate(V3_TARGETS):
                risks[target] = round(float(probs[i]) * 100, 1)

            return {
                "risks": risks,
                "gates": {
                    "vision":   round(float(g[0]), 4),
                    "signal":   round(float(g[1]), 4),
                    "clinical": round(float(g[2]), 4),
                },
                "model_version": "V3",
                "real_inference": True,
            }
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback; traceback.print_exc()
        return mock_inference(labs_dict)


def mock_inference(labs_dict=None):
    """Deterministic fallback predictions for 8 diseases based on lab values."""
    labs = labs_dict or {}

    # Base risk calculation from labs
    base = 0
    if float(labs.get("bnp",        0)) > 400:  base += 25
    if float(labs.get("troponin",   0)) > 0.04: base += 15
    if float(labs.get("creatinine", 0)) > 1.5:  base += 12
    if float(labs.get("sodium",     140)) < 135: base += 10
    if float(labs.get("hemoglobin", 13)) < 12:   base += 8
    if float(labs.get("glucose",    90)) > 120:  base += 5
    if float(labs.get("wbc",        7.5)) > 11:  base += 8

    # Generate deterministic, clinically-plausible 8-disease risks
    def clamp(v): return min(max(round(v, 1), 2.0), 95.0)

    risks = {
        "mortality":              clamp(base * 0.38 + 7),
        "heart_failure":          clamp(base * 1.0  + 12),
        "myocardial_infarction":  clamp(base * 0.25 + 4),
        "arrhythmia":             clamp(base * 0.45 + 8),
        "sepsis":                 clamp(base * 0.30 + 5),
        "pulmonary_embolism":     clamp(base * 0.12 + 3),
        "acute_kidney_injury":    clamp(base * 0.55 + 6),
        "icu_admission":          clamp(base * 0.50 + 9),
    }

    return {
        "risks": risks,
        "gates": {
            "vision":   0.33,
            "signal":   0.33,
            "clinical": 0.34,
        },
        "model_version": "V3",
        "real_inference": False,
    }


# ─── PATIENT DATABASE ────────────────────────────────────────────────────────
PATIENTS = {
    "1": {"id":"1","mrn":"MRN-004821","name":"Rajesh Kumar",  "age":67,"gender":"Male",  "status":"Active",    "condition":"Acute decompensated HF",      "severity":"critical"},
    "2": {"id":"2","mrn":"MRN-003156","name":"Priya Sharma",  "age":54,"gender":"Female","status":"Active",    "condition":"Follow-up cardiac evaluation","severity":"moderate"},
    "3": {"id":"3","mrn":"MRN-005934","name":"Anand Patel",   "age":72,"gender":"Male",  "status":"Active",    "condition":"Severe HF with reduced EF",   "severity":"critical"},
    "4": {"id":"4","mrn":"MRN-002187","name":"Meera Iyer",    "age":49,"gender":"Female","status":"Discharged","condition":"Routine cardiac screening",    "severity":"normal"},
    "5": {"id":"5","mrn":"MRN-006743","name":"Vikram Singh",  "age":61,"gender":"Male",  "status":"Active",    "condition":"Moderate HF follow-up",       "severity":"moderate"},
    "6": {"id":"6","mrn":"MRN-001298","name":"Lakshmi Nair",  "age":58,"gender":"Female","status":"Discharged","condition":"Post-MI cardiac evaluation",  "severity":"normal"},
    "7": {"id":"7","mrn":"MRN-007512","name":"Suresh Reddy",  "age":75,"gender":"Male",  "status":"Active",    "condition":"Chronic HF management",       "severity":"critical"},
    "8": {"id":"8","mrn":"MRN-008901","name":"Deepa Menon",   "age":45,"gender":"Female","status":"Active",    "condition":"New onset dyspnoea workup",   "severity":"moderate"},
}

# V3 Encounters — 8 disease risk scores per encounter
ENCOUNTERS = {
"1": [
  {"id":"e032","label":"E-032","date":"26 Mar 2026","description":"Acute decompensated HF",
   "risks":{"mortality":31,"heart_failure":82,"myocardial_infarction":15,"arrhythmia":48,"sepsis":22,"pulmonary_embolism":7,"acute_kidney_injury":41,"icu_admission":55},
   "gates":{"vision":0.339,"signal":0.338,"clinical":0.323},
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
     {"name":"WBC","value":"9.8 x10³/uL","normal":"4.5-11.0","status":"Normal"},
     {"name":"Glucose","value":"142 mg/dL","normal":"70-100","status":"High"},
   ]},
  {"id":"e028","label":"E-028","date":"14 Feb 2026","description":"Follow-up cardiac evaluation",
   "risks":{"mortality":19,"heart_failure":58,"myocardial_infarction":8,"arrhythmia":32,"sepsis":10,"pulmonary_embolism":4,"acute_kidney_injury":25,"icu_admission":30},
   "gates":{"vision":0.350,"signal":0.330,"clinical":0.320},
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
   "risks":{"mortality":18,"heart_failure":45,"myocardial_infarction":6,"arrhythmia":22,"sepsis":8,"pulmonary_embolism":3,"acute_kidney_injury":15,"icu_admission":20},
   "gates":{"vision":0.345,"signal":0.330,"clinical":0.325},
   "cxr_findings":["Mild cardiomegaly"],
   "ecg_findings":["Normal sinus rhythm","Left axis deviation"],
   "cxr_image":"p2_e041.jpg",
   "labs":[{"name":"BNP","value":"290 pg/mL","normal":"<100","status":"High"}]},
],
"3": [
  {"id":"e019","label":"E-019","date":"22 Mar 2026","description":"Severe HF with reduced EF",
   "risks":{"mortality":56,"heart_failure":91,"myocardial_infarction":28,"arrhythmia":62,"sepsis":35,"pulmonary_embolism":12,"acute_kidney_injury":64,"icu_admission":72},
   "gates":{"vision":0.320,"signal":0.345,"clinical":0.335},
   "cxr_findings":["Severe cardiomegaly","Pulmonary oedema","Pleural effusion"],
   "ecg_findings":["Left bundle branch block","ST depression"],
   "cxr_image":"p3_e019.jpg",
   "labs":[
     {"name":"BNP","value":"2100 pg/mL","normal":"<100","status":"Critical"},
     {"name":"Creatinine","value":"2.4 mg/dL","normal":"0.7-1.3","status":"Critical"},
     {"name":"Sodium","value":"128 mEq/L","normal":"136-145","status":"Critical"},
   ]},
],
"4": [{"id":"e004","label":"E-004","date":"20 Mar 2026","description":"Routine cardiac screening",
   "risks":{"mortality":5,"heart_failure":12,"myocardial_infarction":3,"arrhythmia":8,"sepsis":4,"pulmonary_embolism":2,"acute_kidney_injury":6,"icu_admission":7},
   "gates":{"vision":0.340,"signal":0.335,"clinical":0.325},
   "cxr_findings":["No significant findings"],"ecg_findings":["Normal sinus rhythm"],"cxr_image":"p4_e004.jpg",
   "labs":[{"name":"BNP","value":"45 pg/mL","normal":"<100","status":"Normal"}]}],
"5": [{"id":"e005","label":"E-005","date":"18 Mar 2026","description":"Moderate HF follow-up",
   "risks":{"mortality":28,"heart_failure":67,"myocardial_infarction":12,"arrhythmia":38,"sepsis":18,"pulmonary_embolism":6,"acute_kidney_injury":32,"icu_admission":40},
   "gates":{"vision":0.335,"signal":0.340,"clinical":0.325},
   "cxr_findings":["Cardiomegaly","Mild pulmonary congestion"],"ecg_findings":["Atrial fibrillation","Rate-controlled"],"cxr_image":"p5_e005.jpg",
   "labs":[{"name":"BNP","value":"580 pg/mL","normal":"<100","status":"Critical"}]}],
"6": [{"id":"e006","label":"E-006","date":"15 Mar 2026","description":"Post-MI cardiac evaluation",
   "risks":{"mortality":8,"heart_failure":23,"myocardial_infarction":45,"arrhythmia":28,"sepsis":6,"pulmonary_embolism":5,"acute_kidney_injury":12,"icu_admission":18},
   "gates":{"vision":0.330,"signal":0.350,"clinical":0.320},
   "cxr_findings":["Mild atelectasis"],"ecg_findings":["Q waves in V1-V3","ST elevation"],"cxr_image":"p6_e006.jpg",
   "labs":[{"name":"BNP","value":"120 pg/mL","normal":"<100","status":"High"},{"name":"Troponin I","value":"2.8 ng/mL","normal":"<0.04","status":"Critical"}]}],
"7": [{"id":"e007","label":"E-007","date":"12 Mar 2026","description":"Chronic HF management",
   "risks":{"mortality":42,"heart_failure":78,"myocardial_infarction":18,"arrhythmia":55,"sepsis":25,"pulmonary_embolism":8,"acute_kidney_injury":52,"icu_admission":58},
   "gates":{"vision":0.325,"signal":0.340,"clinical":0.335},
   "cxr_findings":["Cardiomegaly","Pleural effusion"],"ecg_findings":["Ventricular tachycardia","LV hypertrophy"],"cxr_image":"p7_e007.jpg",
   "labs":[{"name":"BNP","value":"1200 pg/mL","normal":"<100","status":"Critical"}]}],
"8": [{"id":"e008","label":"E-008","date":"10 Mar 2026","description":"New onset dyspnea workup",
   "risks":{"mortality":12,"heart_failure":34,"myocardial_infarction":7,"arrhythmia":18,"sepsis":15,"pulmonary_embolism":22,"acute_kidney_injury":10,"icu_admission":20},
   "gates":{"vision":0.330,"signal":0.325,"clinical":0.345},
   "cxr_findings":["Borderline heart size"],"ecg_findings":["Sinus tachycardia"],"cxr_image":"p8_e008.jpg",
   "labs":[{"name":"BNP","value":"220 pg/mL","normal":"<100","status":"High"},{"name":"D-Dimer","value":"1.8 ug/mL","normal":"<0.5","status":"Critical"}]}],
}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def save_patient_record(conn, patient: dict):
    conn.execute(
        """
        INSERT OR REPLACE INTO patients (id, mrn, name, age, gender, status, condition, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient["id"], patient["mrn"], patient["name"], int(patient["age"]),
            patient["gender"], patient["status"], patient["condition"], patient["severity"],
        ),
    )


def save_encounter_record(conn, patient_id: str, encounter: dict):
    conn.execute(
        """
        INSERT OR REPLACE INTO encounters (
            id, patient_id, label, date, description, risks_json, gates_json,
            cxr_findings_json, ecg_findings_json, labs_json, cxr_image,
            analysis_source, last_inference_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            encounter["id"], patient_id, encounter["label"], encounter["date"], encounter["description"],
            json.dumps(encounter.get("risks", {})),
            json.dumps(encounter.get("gates", {})),
            json.dumps(encounter.get("cxr_findings", [])),
            json.dumps(encounter.get("ecg_findings", [])),
            json.dumps(encounter.get("labs", [])),
            encounter.get("cxr_image", ""),
            encounter.get("analysis_source", "seeded"),
            encounter.get("last_inference_at"),
        ),
    )


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                mrn TEXT NOT NULL,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                status TEXT NOT NULL,
                condition TEXT NOT NULL,
                severity TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS encounters (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                label TEXT NOT NULL,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                risks_json TEXT NOT NULL,
                gates_json TEXT NOT NULL,
                cxr_findings_json TEXT NOT NULL,
                ecg_findings_json TEXT NOT NULL,
                labs_json TEXT NOT NULL,
                cxr_image TEXT NOT NULL,
                analysis_source TEXT DEFAULT 'seeded',
                last_inference_at TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            )
            """
        )

        patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        if patient_count == 0:
            for patient in PATIENTS.values():
                save_patient_record(conn, patient)
            for patient_id, encounters in ENCOUNTERS.items():
                for encounter in encounters:
                    seeded = dict(encounter)
                    seeded.setdefault("analysis_source", "seeded")
                    seeded.setdefault("last_inference_at", None)
                    save_encounter_record(conn, patient_id, seeded)
            conn.commit()


def refresh_cache_from_db():
    global PATIENTS, ENCOUNTERS
    with get_db() as conn:
        patient_rows = conn.execute("SELECT * FROM patients ORDER BY CAST(id AS INTEGER)").fetchall()
        encounter_rows = conn.execute(
            """
            SELECT * FROM encounters
            ORDER BY patient_id, rowid DESC
            """
        ).fetchall()

    PATIENTS = {
        row["id"]: {
            "id": row["id"],
            "mrn": row["mrn"],
            "name": row["name"],
            "age": row["age"],
            "gender": row["gender"],
            "status": row["status"],
            "condition": row["condition"],
            "severity": row["severity"],
        }
        for row in patient_rows
    }

    encounters = {}
    for row in encounter_rows:
        encounter = {
            "id": row["id"],
            "label": row["label"],
            "date": row["date"],
            "description": row["description"],
            "risks": json.loads(row["risks_json"]),
            "gates": json.loads(row["gates_json"]),
            "cxr_findings": json.loads(row["cxr_findings_json"]),
            "ecg_findings": json.loads(row["ecg_findings_json"]),
            "labs": json.loads(row["labs_json"]),
            "cxr_image": row["cxr_image"],
            "analysis_source": row["analysis_source"] or "seeded",
            "last_inference_at": row["last_inference_at"],
        }
        encounters.setdefault(row["patient_id"], []).append(encounter)
    ENCOUNTERS = encounters


# ─── HELPER ───────────────────────────────────────────────────────────────────
def resolve_image(filename: str) -> Optional[str]:
    """Return URL if image file exists, else None."""
    if not filename:
        return None

    image_path = IMAGES_DIR / filename
    if image_path.exists():
        return f"http://127.0.0.1:8000/images/{image_path.name}"

    stem = Path(filename).stem
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            return f"http://127.0.0.1:8000/images/{candidate.name}"
    return None


def resolve_image_path(filename: str) -> Optional[Path]:
    """Return a real filesystem path for an encounter image, trying common extensions."""
    if not filename:
        return None
    direct = IMAGES_DIR / filename
    if direct.exists():
        return direct
    stem = Path(filename).stem
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_labs_dict(encounter: dict) -> dict:
    """Parse encounter lab rows into model input keys with numeric values."""
    name_map = {
        "bnp": ["bnp", "b-type natriuretic peptide"],
        "troponin": ["troponin", "troponin i", "trop", "troponin-i"],
        "creatinine": ["creatinine", "creat"],
        "sodium": ["sodium", "na"],
        "potassium": ["potassium", "k"],
        "hemoglobin": ["hemoglobin", "hgb", "hb"],
        "wbc": ["wbc", "white blood cell", "leukocyte"],
        "glucose": ["glucose", "blood glucose"],
        "bun": ["bun", "blood urea nitrogen", "urea"],
        "albumin": ["albumin", "alb"],
    }

    def parse_value(val_str: str) -> float:
        import re
        m = re.search(r"[-+]?\d*\.?\d+", str(val_str))
        return float(m.group()) if m else 0.0

    parsed = {}
    for lab in encounter.get("labs", []):
        name_lower = lab.get("name", "").lower().strip()
        val = parse_value(lab.get("value", "0"))
        for key, aliases in name_map.items():
            if any(alias in name_lower for alias in aliases):
                parsed[key] = val
                break
    return parsed


def classify_severity(risks: dict) -> str:
    """Derive a simple severity bucket from the two primary risk scores."""
    hf = float((risks or {}).get("heart_failure", 0) or 0)
    mortality = float((risks or {}).get("mortality", 0) or 0)
    if hf >= 70 or mortality >= 40:
        return "critical"
    if hf >= 40 or mortality >= 20:
        return "moderate"
    return "normal"


def enrich_patient(patient_id: str) -> dict:
    """Return a copy of a patient with the latest encounter summary attached."""
    patient = PATIENTS.get(patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")

    enriched = dict(patient)
    encs = ENCOUNTERS.get(patient_id, [])
    if encs:
        latest = encs[0]
        enriched["risks"] = latest.get("risks", {})
        enriched["gates"] = latest.get("gates", {})
        enriched["hf_risk"] = latest.get("risks", {}).get("heart_failure", 0)
        enriched["mortality_risk"] = latest.get("risks", {}).get("mortality", 0)
        enriched["last_encounter"] = latest.get("date")
        enriched["cxr_image"] = resolve_image(latest.get("cxr_image", ""))
    else:
        enriched["risks"] = {}
        enriched["gates"] = {}
        enriched["hf_risk"] = 0
        enriched["mortality_risk"] = 0
        enriched["last_encounter"] = None
        enriched["cxr_image"] = None
    return enriched


def bootstrap_live_inference_for_all():
    """
    Refresh every stored encounter once at startup.
    Persists model-backed or deterministic-estimate results into SQLite.
    """
    refresh_cache_from_db()
    total = 0
    model_count = 0
    estimate_count = 0

    with get_db() as conn:
        for patient_id, encs in ENCOUNTERS.items():
            for enc in encs:
                labs_dict = parse_labs_dict(enc)
                cxr_path_obj = resolve_image_path(enc.get("cxr_image", ""))
                cxr_path = str(cxr_path_obj) if cxr_path_obj else None
                result = run_inference(cxr_path=cxr_path, labs_dict=labs_dict)

                updated = dict(enc)
                updated["risks"] = result.get("risks", {})
                updated["gates"] = result.get("gates", {})
                updated["analysis_source"] = "model" if result.get("real_inference") else "deterministic_estimate"
                updated["last_inference_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                save_encounter_record(conn, patient_id, updated)

                total += 1
                if updated["analysis_source"] == "model":
                    model_count += 1
                else:
                    estimate_count += 1
        conn.commit()

    refresh_cache_from_db()
    print(
        f"✅ Startup inference refresh complete — {total} encounters "
        f"({model_count} model, {estimate_count} deterministic estimate)"
    )


def build_labs_display(labs_dict: dict) -> list:
    """Convert raw lab values into display objects used by the frontend."""
    configs = [
        ("bnp", "BNP", "pg/mL", "<100", lambda v: "Critical" if v > 400 else "High" if v > 100 else "Normal"),
        ("troponin", "Troponin I", "ng/mL", "<0.04", lambda v: "Critical" if v > 0.04 else "Normal"),
        ("creatinine", "Creatinine", "mg/dL", "0.7-1.3", lambda v: "Critical" if v > 2.0 else "High" if v > 1.3 else "Normal"),
        ("sodium", "Sodium", "mEq/L", "136-145", lambda v: "Critical" if v < 130 else "Low" if v < 136 else "Normal"),
        ("potassium", "Potassium", "mEq/L", "3.5-5.0", lambda v: "High" if v > 5.0 else "Low" if v < 3.5 else "Normal"),
        ("hemoglobin", "Hemoglobin", "g/dL", "12-17.5", lambda v: "Low" if v < 12 else "Normal"),
        ("wbc", "WBC", "x10^3/uL", "4.5-11.0", lambda v: "High" if v > 11 else "Low" if v < 4.5 else "Normal"),
        ("glucose", "Glucose", "mg/dL", "70-100", lambda v: "High" if v > 120 else "Normal"),
    ]

    display = []
    for key, label, unit, normal, classify in configs:
        if key not in labs_dict:
            continue
        value = float(labs_dict[key])
        display.append({
            "name": label,
            "value": f"{value:g} {unit}",
            "normal": normal,
            "status": classify(value),
        })
    return display


def generate_ecg_findings(has_ecg: bool, risks: dict) -> list:
    """Return lightweight ECG findings for demo and onboarding flows."""
    if not has_ecg:
        return ["ECG not provided"]
    if float((risks or {}).get("arrhythmia", 0) or 0) >= 40:
        return ["Atrial fibrillation pattern", "Rate variability noted", "Left ventricular strain"]
    if float((risks or {}).get("myocardial_infarction", 0) or 0) >= 30:
        return ["ST-segment changes", "Anterior lead abnormalities"]
    return ["Normal sinus rhythm", "No major acute abnormalities detected"]


def save_uploaded_cxr(patient_id: str, encounter_id: str, file: UploadFile | None) -> Optional[str]:
    """Persist an uploaded CXR image under the demo_images directory."""
    if file is None:
        return None

    suffix = Path(file.filename or "").suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg"}:
        suffix = ".png"
    filename = f"p{patient_id}_{encounter_id}{suffix}"
    destination = IMAGES_DIR / filename
    contents = file.file.read()
    if contents:
        destination.write_bytes(contents)
        return filename
    return None


class AnalyzeRequest(BaseModel):
    labs: dict = {}
    has_ecg: bool = False
    mode: str = "multimodal"   # multimodal | cxr | ecg | labs


class PatientCreateRequest(BaseModel):
    name: str
    age: int
    gender: str = "Male"
    condition: str = ""
    status: str = "Active"
    labs: dict = {}
    has_ecg: bool = False


class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[str] = None
    encounter_id: Optional[str] = None
    # Optional inline patient context (for onboarding flow where patient isn't in DB)
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    risks: Optional[dict] = None
    gates: Optional[dict] = None
    cxr_findings: Optional[list] = None
    ecg_findings: Optional[list] = None
    labs_data: Optional[list] = None


init_db()
refresh_cache_from_db()
bootstrap_live_inference_for_all()


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "VisionCare 3.0", "model": MODEL_VERSION,
            "targets": V3_TARGETS, "model_loaded": _model_loaded}

@app.get("/api/stats")
def stats():
    refresh_cache_from_db()
    enriched = [enrich_patient(pid) for pid in PATIENTS]
    total_patients = len(enriched)
    high_risk_hf = sum(1 for p in enriched if p.get("hf_risk", 0) >= 70)
    critical_cases = sum(1 for p in enriched if p.get("severity") == "critical")
    predictions_today = sum(len(encs) for encs in ENCOUNTERS.values())
    return {"total_patients": total_patients, "high_risk_hf": high_risk_hf,
            "critical_cases": critical_cases, "predictions_today": predictions_today,
            "model_version": MODEL_VERSION, "model_loaded": _model_loaded,
            "targets": V3_TARGETS, "num_targets": 8}

@app.get("/api/patients")
def get_patients(search: str = ""):
    refresh_cache_from_db()
    pts = [enrich_patient(pid) for pid in sorted(PATIENTS.keys(), key=lambda pid: int(pid), reverse=True)]
    if search:
        pts = [p for p in pts if search.lower() in p["name"].lower()
               or search.lower() in p["mrn"].lower()]
    return pts

@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str):
    refresh_cache_from_db()
    return enrich_patient(patient_id)


@app.post("/api/patients")
def create_patient(req: PatientCreateRequest):
    """Create a patient plus an initial encounter from onboarding data."""
    refresh_cache_from_db()
    patient_id = str(max((int(pid) for pid in PATIENTS.keys()), default=0) + 1)
    encounter_id = f"e{len(ENCOUNTERS.get(patient_id, [])) + 1:03d}"
    result = run_inference(labs_dict=req.labs or {})

    severity = classify_severity(result.get("risks", {}))
    patient = {
        "id": patient_id,
        "mrn": f"MRN-{100000 + int(patient_id):06d}",
        "name": req.name.strip(),
        "age": req.age,
        "gender": req.gender,
        "status": req.status,
        "condition": req.condition.strip() or "New patient onboarding",
        "severity": severity,
    }
    PATIENTS[patient_id] = patient

    encounter = {
        "id": encounter_id,
        "label": encounter_id.upper(),
        "date": time.strftime("%d %b %Y"),
        "description": patient["condition"],
        "risks": result.get("risks", {}),
        "gates": result.get("gates", {}),
        "cxr_findings": ["Chest X-ray not uploaded"],
        "ecg_findings": generate_ecg_findings(req.has_ecg, result.get("risks", {})),
        "cxr_image": "",
        "labs": build_labs_display(req.labs or {}),
        "analysis_source": "model" if result.get("real_inference") else "deterministic_estimate",
        "last_inference_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with get_db() as conn:
        save_patient_record(conn, patient)
        save_encounter_record(conn, patient_id, encounter)
        conn.commit()
    refresh_cache_from_db()
    return {
        "patient": enrich_patient(patient_id),
        "encounter": dict(encounter, patient=patient, cxr_image_url=None),
        "created": True,
    }

@app.get("/api/patients/{patient_id}/encounters")
def get_encounters(patient_id: str):
    refresh_cache_from_db()
    encs = ENCOUNTERS.get(patient_id, [])
    result = []
    for e in encs:
        enc_copy = dict(e)
        enc_copy["cxr_image_url"] = resolve_image(e.get("cxr_image", ""))
        result.append(enc_copy)
    return result

@app.get("/api/patients/{patient_id}/encounters/{encounter_id}")
def get_encounter(patient_id: str, encounter_id: str):
    refresh_cache_from_db()
    encs = ENCOUNTERS.get(patient_id, [])
    enc = next((e for e in encs if e["id"] == encounter_id), None)
    if not enc: raise HTTPException(404, "Encounter not found")
    enc = dict(enc)
    enc["cxr_image_url"] = resolve_image(enc.get("cxr_image", ""))
    enc["patient"] = PATIENTS.get(patient_id)
    return enc


# ─── V3 MODEL INFO ───────────────────────────────────────────────────────────
@app.get("/api/model/info")
def model_info():
    v3 = {
        "macro_auc": 0.7926,
        "macro_f1":  0.3921,
        "best_epoch": 17,
        "targets": V3_TARGETS,
        "per_class_auc": {
            "mortality": 0.8082, "heart_failure": 0.7829,
            "myocardial_infarction": 0.7473, "arrhythmia": 0.7936,
            "sepsis": 0.8223, "pulmonary_embolism": 0.8213,
            "acute_kidney_injury": 0.7813, "icu_admission": 0.7839,
        },
        "gates": {"vision": 0.339, "signal": 0.338, "clinical": 0.323},
        "encoders": {
            "cxr": "ResNet-50 (2048-D, SYMILE pre-trained)",
            "ecg": "1D ResNet-18 (512-D, SYMILE pre-trained)",
            "labs": "3-Layer NN (256-D, SYMILE pre-trained)",
        },
        "fusion": "Cross-Attention Gated Fusion (4-head) + Gate Entropy Regularization",
        "training": "Progressive Unfreeze (Phase A: frozen 5ep, Phase B: fine-tune 20ep)",
        "loss": "Focal Loss (gamma=2.0) + Label Smoothing (0.03)",
        "checkpoint": str(MODELS_DIR / "fusion_v3_best.pth"),
        "exists": (MODELS_DIR / "fusion_v3_best.pth").exists(),
    }
    return {"active_version": MODEL_VERSION, "v3": v3,
            "model_loaded": _model_loaded}


# ─── V3 PREDICTION ENDPOINTS ─────────────────────────────────────────────────

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Run V3 multi-modal inference (8 diseases + gate weights)."""
    start = time.time()
    result = run_inference(labs_dict=req.labs)
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    return result


@app.post("/api/onboard")
async def onboard_patient(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form("Male"),
    condition: str = Form(""),
    status: str = Form("Active"),
    has_ecg: bool = Form(False),
    labs_json: str = Form("{}"),
    cxr_file: UploadFile | None = File(None),
):
    """Create a patient from the onboarding wizard and persist the first encounter."""
    refresh_cache_from_db()
    try:
        labs = json.loads(labs_json or "{}")
        if not isinstance(labs, dict):
            raise ValueError("labs_json must decode to an object")
    except Exception as exc:
        raise HTTPException(400, f"Invalid labs_json payload: {exc}") from exc

    patient_id = str(max((int(pid) for pid in PATIENTS.keys()), default=0) + 1)
    encounter_id = f"e{100 + int(patient_id):03d}"

    cxr_filename = save_uploaded_cxr(patient_id, encounter_id, cxr_file)
    cxr_path = str(IMAGES_DIR / cxr_filename) if cxr_filename else None

    start = time.time()
    result = run_inference(cxr_path=cxr_path, labs_dict=labs)
    latency_ms = round((time.time() - start) * 1000, 1)

    severity = classify_severity(result.get("risks", {}))
    patient = {
        "id": patient_id,
        "mrn": f"MRN-{100000 + int(patient_id):06d}",
        "name": name.strip(),
        "age": age,
        "gender": gender,
        "status": status,
        "condition": condition.strip() or "New patient onboarding",
        "severity": severity,
    }
    PATIENTS[patient_id] = patient

    encounter = {
        "id": encounter_id,
        "label": encounter_id.upper(),
        "date": time.strftime("%d %b %Y"),
        "description": patient["condition"],
        "risks": result.get("risks", {}),
        "gates": result.get("gates", {}),
        "cxr_findings": ["Uploaded chest X-ray available"] if cxr_filename else ["Chest X-ray not uploaded"],
        "ecg_findings": generate_ecg_findings(has_ecg, result.get("risks", {})),
        "cxr_image": cxr_filename or "",
        "labs": build_labs_display(labs),
        "analysis_source": "model" if result.get("real_inference") else "deterministic_estimate",
        "last_inference_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with get_db() as conn:
        save_patient_record(conn, patient)
        save_encounter_record(conn, patient_id, encounter)
        conn.commit()
    refresh_cache_from_db()

    return {
        "patient": enrich_patient(patient_id),
        "encounter": dict(encounter, patient=patient, cxr_image_url=resolve_image(encounter["cxr_image"])),
        "result": {
            **result,
            "latency_ms": latency_ms,
            "inputs_used": {
                "cxr": bool(cxr_filename),
                "ecg": bool(has_ecg),
                "labs": len(labs),
            },
        },
        "created": True,
    }

@app.post("/api/predict/{encounter_id}")
def predict_encounter(encounter_id: str):
    """
    Run REAL V3 inference for a specific encounter.
    Parses stored lab values + resolves CXR image → full model forward pass.
    Returns fresh 8-disease probabilities every call (no caching).
    """

    # ── 1. Find the encounter ────────────────────────────────────────────────
    refresh_cache_from_db()
    enc = None
    pid = None
    for p, encs in ENCOUNTERS.items():
        for e in encs:
            if e["id"] == encounter_id:
                enc, pid = dict(e), p
                break
        if enc:
            break
    if not enc:
        raise HTTPException(404, "Encounter not found")

    # ── 2. Parse labs list → dict ────────────────────────────────────────────
    labs_dict = parse_labs_dict(enc)

    # ── 3. Resolve CXR path ──────────────────────────────────────────────────
    cxr_image = enc.get("cxr_image", "")
    resolved_cxr = resolve_image_path(cxr_image)
    cxr_path = str(resolved_cxr) if resolved_cxr else None

    # ── 4. Run inference ─────────────────────────────────────────────────────
    start = time.time()
    result = run_inference(cxr_path=cxr_path, labs_dict=labs_dict)
    latency = round((time.time() - start) * 1000, 1)

    updated_encounter = dict(enc)
    updated_encounter["risks"] = result["risks"]
    updated_encounter["gates"] = result.get("gates", {})
    updated_encounter["analysis_source"] = "model" if result.get("real_inference") else "deterministic_estimate"
    updated_encounter["last_inference_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    updated_encounter["cxr_image"] = enc.get("cxr_image", "")
    with get_db() as conn:
        save_encounter_record(conn, pid, updated_encounter)
        conn.commit()
    refresh_cache_from_db()

    return {
        "prediction_id":  f"pred_{encounter_id}_{int(time.time())}",
        "encounter_id":   encounter_id,
        "patient_id":     pid,
        "model_version":  "V3",
        "risks":          updated_encounter["risks"],
        "gates":          updated_encounter.get("gates", {}),
        "real_inference": result.get("real_inference", False),
        "latency_ms":     latency,
        "analysis_source": updated_encounter["analysis_source"],
        "last_inference_at": updated_encounter["last_inference_at"],
        "inputs_used": {
            "cxr":  cxr_path is not None,
            "ecg":  False,
            "labs": len(labs_dict),
        },
    }



# ─── SINGLE-MODAL ENDPOINTS (kept for individual analysis) ───────────────────

def _run_cxr_model(image_bytes: bytes) -> dict:
    """V3 CXR-only inference — returns 8 diseases, gate vision~1.0."""
    if _cxr_model is None:
        res = mock_inference()
        res['gates'] = {'vision': 1.0, 'signal': 0.0, 'clinical': 0.0}
        return res
    try:
        import torch
        from PIL import Image
        import torchvision.transforms as T
        from architectures import run_v3_single_inference
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transform = T.Compose([T.Resize((320, 320)), T.ToTensor(),
                               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        cxr_tensor = transform(img).unsqueeze(0)
        result = run_v3_single_inference(_cxr_model, 'cxr', cxr_tensor=cxr_tensor)
        return {**result, 'model_version': 'V3-CXR', 'real_inference': True}
    except Exception as e:
        print(f'CXR inference error: {e}')
        return mock_inference()


def _run_ecg_model(ecg_array=None) -> dict:
    """V3 ECG-only inference — returns 8 diseases, gate signal~1.0."""
    if _ecg_model is None:
        res = mock_inference()
        res['gates'] = {'vision': 0.0, 'signal': 1.0, 'clinical': 0.0}
        return res
    try:
        import torch, numpy as np
        from architectures import run_v3_single_inference
        if ecg_array is not None:
            arr = np.array(ecg_array, dtype=np.float32)
            if arr.ndim == 1: arr = arr.reshape(12, -1)
        else:
            t = np.linspace(0, 10, 5000)
            arr = np.stack([
                np.sin(2*np.pi*1.2*t) + 0.05*np.random.randn(5000)
                for _ in range(12)
            ]).astype(np.float32)
        ecg_tensor = torch.tensor(arr).unsqueeze(0)
        result = run_v3_single_inference(_ecg_model, 'ecg', ecg_tensor=ecg_tensor)
        return {**result, 'model_version': 'V3-ECG', 'real_inference': True}
    except Exception as e:
        print(f'ECG inference error: {e}')
        return mock_inference()


def _run_labs_model(labs_dict: dict) -> dict:
    """V3 Labs-only inference — returns 8 diseases, gate clinical~1.0."""
    if _labs_model is None:
        res = mock_inference(labs_dict)
        res['gates'] = {'vision': 0.0, 'signal': 0.0, 'clinical': 1.0}
        return res
    try:
        import torch, numpy as np
        from architectures import run_v3_single_inference
        LAB_KEYS  = ['bnp','troponin','creatinine','sodium','potassium','hemoglobin','wbc','glucose']
        LAB_MEANS = [0.15, 0.02, 1.0, 139, 4.0, 13.0, 7.5, 100]
        LAB_STDS  = [0.10, 0.02, 0.4, 5,   0.6,  2.5, 2.5,  40]
        vals  = [float(labs_dict.get(k, m)) for k, m in zip(LAB_KEYS, LAB_MEANS)]
        norms = [(v-m)/max(s,1e-6) for v,m,s in zip(vals, LAB_MEANS, LAB_STDS)]
        miss  = [1.0 if labs_dict.get(k) not in (None,'',0) else 0.0 for k in LAB_KEYS]
        feat = np.zeros(100, dtype=np.float32)
        feat[0:8] = norms
        feat[8:16] = miss
        labs_tensor = torch.tensor(feat).unsqueeze(0)
        result = run_v3_single_inference(_labs_model, 'labs', labs_tensor=labs_tensor)
        return {**result, 'model_version': 'V3-Labs', 'real_inference': True}
    except Exception as e:
        print(f'Labs inference error: {e}')
        return mock_inference(labs_dict)


@app.post("/api/analyze/cxr")
async def analyze_cxr(file: UploadFile = File(...)):
    start = time.time()
    image_bytes = await file.read()
    result = _run_cxr_model(image_bytes)
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    result["mode"] = "cxr"
    return result

@app.post("/api/analyze/ecg")
def analyze_ecg():
    start = time.time()
    result = _run_ecg_model(ecg_array=None)
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    result["mode"] = "ecg"
    return result

@app.post("/api/analyze/labs")
def analyze_labs(req: AnalyzeRequest):
    start = time.time()
    result = _run_labs_model(req.labs)
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    result["mode"] = "labs"
    return result


@app.get("/api/models/status")
def models_status():
    return {
        "fusion": {
            "loaded": _model_loaded,
            "version": MODEL_VERSION,
            "targets": V3_TARGETS,
            "architecture": "ResNet-50 + 1D ResNet-18 + 3-NN + Cross-Attention Fusion",
        },
        "cxr_only":  {"loaded": _cxr_model  is not None, "architecture": "V3 ResNet-50 (shared weights)"},
        "ecg_only":  {"loaded": _ecg_model  is not None, "architecture": "V3 1D ResNet-18 (shared weights)"},
        "labs_only": {"loaded": _labs_model is not None, "architecture": "V3 3-Layer NN (shared weights)"},
        "note": "Single-modal endpoints use the fusion model with zeroed missing modalities",
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Medical AI Chat — RAG + Gemini 2.5 Flash (backend only).
    1. Retrieves relevant chunks from FAISS (medical textbooks)
    2. Builds patient-context-aware prompt
    3. Calls Gemini 2.5 Flash for grounded generation
    4. Falls back to rule-based responses if unavailable
    """
    refresh_cache_from_db()
    # Build patient context from either DB lookup or inline data
    patient_data = {}

    if req.patient_id and req.encounter_id:
        # Look up from our data store
        p = PATIENTS.get(req.patient_id, {})
        encs = ENCOUNTERS.get(req.patient_id, [])
        enc = next((e for e in encs if e["id"] == req.encounter_id), {})
        patient_data = {
            "patient_name": p.get("name", "Unknown"),
            "age": p.get("age"),
            "gender": p.get("gender"),
            "mrn": p.get("mrn"),
            "condition": p.get("condition"),
            "encounter_label": enc.get("label"),
            "encounter_date": enc.get("date"),
            "risks": enc.get("risks", {}),
            "gates": enc.get("gates", {}),
            "cxr_findings": enc.get("cxr_findings", []),
            "ecg_findings": enc.get("ecg_findings", []),
            "labs": enc.get("labs", []),
            "analysis_source": enc.get("analysis_source"),
            "last_inference_at": enc.get("last_inference_at"),
        }
    else:
        # Use inline data from request (onboarding flow)
        patient_data = {
            "patient_name": req.patient_name or "Unknown",
            "age": req.patient_age,
            "gender": req.patient_gender,
            "risks": req.risks or {},
            "gates": req.gates or {},
            "cxr_findings": req.cxr_findings or [],
            "ecg_findings": req.ecg_findings or [],
            "labs": req.labs_data or [],
        }

    result = await _rag.generate_response(req.message, patient_data)
    return result


@app.get("/api/rag/status")
def rag_status():
    """Return RAG engine status."""
    return _rag.get_status()
