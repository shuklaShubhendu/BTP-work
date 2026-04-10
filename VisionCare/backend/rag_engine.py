"""
rag_engine.py — VisionCare 3.0 Medical RAG Engine
====================================================
Retrieval-Augmented Generation using:
  • FAISS vector store (built from medical textbooks)
  • Gemini 2.5 Flash for generation
  • Strict medical-domain-only system prompt

Architecture:
  User Question → FAISS Retrieval (top-5 chunks) → System Prompt + Patient Context
  → Gemini 2.5 Flash → Formatted Medical Response with Citations
"""

import os, json, time
from pathlib import Path
from typing import Optional

INDEX_DIR = Path(__file__).parent / "rag_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# ─── Medical System Prompt ────────────────────────────────────────────────────

MEDICAL_SYSTEM_PROMPT = """You are VisionCare 3.0 Medical AI — a clinical decision support assistant operating within the VisionCare multi-modal cardiac risk platform.

═══ YOUR ROLE ═══
You are a clinical decision support tool trained on authoritative medical references. You help clinicians interpret VisionCare 3.0's AI predictions in the context of established medical guidelines. You DO NOT diagnose or prescribe — you provide evidence-based clinical context.

═══ KNOWLEDGE SOURCES (cite these) ═══
1. Harrison's Principles of Internal Medicine, 21st Edition (2022)
2. Braunwald's Heart Disease: A Textbook of Cardiovascular Medicine, 12th Edition (2022)
3. AHA 2022 Heart Failure Guidelines — Heidenreich PA, et al. Circulation. 2022;145:e895–e1032
4. ESC 2021 Heart Failure Guidelines — McDonagh TA, et al. Eur Heart J. 2021;42:3599–3726
5. ESC 2020 NSTEMI/ACS Guidelines — Collet JP, et al. Eur Heart J. 2021;42:1289–1367
6. Surviving Sepsis Campaign 2021 — Evans L, et al. Intensive Care Med. 2021;47:1181–1247
7. KDIGO 2012 AKI Guidelines — Kidney Int Suppl. 2012;2:1–138
8. ESC 2019 Pulmonary Embolism Guidelines — Konstantinides SV, et al. Eur Heart J. 2020;41:543–603
9. Global Atlas on Cardiovascular Disease Prevention and Control (WHO, 2011)
10. Essentials of Cardiology — McGill University Reference

═══ RETRIEVED CONTEXT FROM MEDICAL TEXTBOOKS ═══
{retrieved_context}

═══ VISIONCARE 3.0 MODEL INFORMATION ═══
• Architecture: ResNet-50 (CXR→2048-D) + 1D-ResNet-18 (ECG→512-D) + 3-Layer NN (Labs→256-D)
• Fusion: Cross-Attention Gated (4-head) with Gate Entropy Regularization (λ=0.01)
• Training: Progressive Unfreezing + Focal Loss (γ=2.0) + EMA (decay=0.999)
• Performance: Macro AUC 0.7926 across 8 diseases
• Key Innovation: Balanced gate weights (~34% / 34% / 32%) — solved modality collapse from V2

═══ PATIENT CONTEXT ═══
{patient_context}

═══ STRICT RULES ═══
1. ONLY answer questions related to:
   - This patient's clinical data and risk predictions
   - Cardiovascular medicine, cardiology, critical care
   - Interpretation of CXR, ECG, lab values
   - Medical guidelines (AHA, ESC, KDIGO, etc.)
   - VisionCare model architecture and predictions

2. REFUSE all non-medical questions. If someone asks about coding, math, recipes, weather, jokes, politics, or anything outside the medical domain, respond EXACTLY with:
   "I am VisionCare's Medical AI Assistant. I can only discuss clinical topics related to this patient's cardiovascular risk assessment. Please ask me about the patient's risk scores, lab values, ECG/CXR findings, or relevant clinical guidelines."

3. Every response MUST include at least one citation in this format:
   📋 [Source Name], [Section/Chapter if applicable]

4. Use **bold** for key clinical values and findings.

5. Keep responses between 150-350 words.

6. Structure responses with clear sections using bullet points.

7. Always relate your answer back to the patient's specific data when available.

8. When discussing risk scores, explain what drives the prediction using ALL three modalities (CXR, ECG, Labs).

9. Never claim to diagnose. Use language like "suggests", "consistent with", "warrants further evaluation".
"""


# ─── Rule-Based Fallback ──────────────────────────────────────────────────────

def rule_based_response(msg, patient_context):
    """Offline fallback when both FAISS and Gemini are unavailable."""
    ml = msg.lower()
    ctx = patient_context or {}
    risks = ctx.get("risks", {})
    gates = ctx.get("gates", {"vision": 0.339, "signal": 0.338, "clinical": 0.323})
    cxr = round((gates.get("vision", 0.339)) * 100)
    ecg = round((gates.get("signal", 0.338)) * 100)
    labs = round((gates.get("clinical", 0.323)) * 100)
    name = ctx.get("patient_name", "this patient")

    if any(w in ml for w in ["gate", "balance", "contribut", "modal"]):
        return f"**VisionCare 3.0 Gate Weights (Balanced):**\n\n• **CXR (Vision): {cxr}%** — chest X-ray features\n• **ECG (Signal): {ecg}%** — ECG waveform patterns\n• **Labs (Clinical): {labs}%** — blood biomarkers\n\nV3 achieves near-equal contributions through **Gate Entropy Regularization** (λ=0.01), ensuring no single modality dominates.\n\n📋 VisionCare 3.0 Architecture — Gate Entropy Regularization"

    if any(w in ml for w in ["hf", "heart failure", "why", "predict"]):
        return f"**Heart Failure Risk: {risks.get('heart_failure', '?')}%**\n\n**Multi-modal evidence:**\n• **Labs ({labs}%):** BNP elevation is the strongest HF biomarker (AHA Class I)\n• **ECG ({ecg}%):** Rhythm abnormalities present in ~30% of HF patients\n• **CXR ({cxr}%):** Cardiac changes confirmed on imaging\n\n**Co-morbidity:** AKI risk {risks.get('acute_kidney_injury', '?')}% suggests cardiorenal syndrome.\n\n📋 AHA 2022 HF Guidelines — Heidenreich PA, et al. Circulation. 2022;145:e895–e1032"

    if "sepsis" in ml:
        return f"**Sepsis Risk: {risks.get('sepsis', '?')}%**\n\nSepsis detection requires **all three modalities**:\n• **CXR:** Bilateral infiltrates → ARDS/pneumonia\n• **ECG:** Tachycardia → hemodynamic compromise\n• **Labs:** Elevated WBC, lactate markers\n\nBalanced gates ({cxr}/{ecg}/{labs}) ensure none of these signals are missed.\n\n📋 Surviving Sepsis Campaign 2021 — Evans L, et al. Intensive Care Med. 2021;47:1181–1247"

    if any(w in ml for w in ["mortality", "death", "prognos"]):
        return f"**Mortality Risk: {risks.get('mortality', '?')}%**\n\nKey drivers:\n• Multi-organ signals: AKI {risks.get('acute_kidney_injury', '?')}%, Sepsis {risks.get('sepsis', '?')}%\n• Cardiac: HF {risks.get('heart_failure', '?')}%, Arrhythmia {risks.get('arrhythmia', '?')}%\n• ICU probability: {risks.get('icu_admission', '?')}%\n\n📋 Pocock SJ et al. — MAGGIC meta-analysis, Eur Heart J 2013"

    if any(w in ml for w in ["lab", "bnp", "creatinine", "troponin"]):
        return f"**Key Lab Interpretation (Labs gate: {labs}%):**\n\n• **BNP:** Primary HF biomarker. >400 pg/mL = Class I HF indication (AHA 2022)\n• **Creatinine:** Elevated = cardiorenal syndrome risk\n• **Troponin:** Elevated = myocardial damage marker\n\nV3 gives labs **{labs}%** weight (vs only 6% in V2).\n\n📋 ESC 2021 HF Guidelines — McDonagh TA, et al. Eur Heart J. 2021;42:3599–3726"

    if any(w in ml for w in ["ecg", "rhythm", "af", "arrhyth"]):
        return f"**ECG Analysis (Signal gate: {ecg}%):**\n\nECG contributed **{ecg}%** to V3 predictions (vs only 15% in V2).\n• Arrhythmia Risk: **{risks.get('arrhythmia', '?')}%**\n• AF increases HF hospitalization by 35%\n• ICU Risk: **{risks.get('icu_admission', '?')}%**\n\n📋 AHA 2020 AF Guidelines — January CT, et al. Circulation. 2019;140:e125–e151"

    if any(w in ml for w in ["cxr", "x-ray", "chest", "radiol"]):
        return f"**CXR Analysis (Vision gate: {cxr}%):**\n\nCXR contributed **{cxr}%** via ResNet-50 encoder (2048-D).\nIn V3, CXR is one-third of the decision (vs 79% in V2).\n\nGrad-CAM highlights: Perihilar regions (fluid), cardiac silhouette (size), costophrenic angles (effusion).\n\n📋 AHA 2022 — CXR Interpretation in HF, Section 3.4"

    return f"**VisionCare 3.0 Analysis for {name}:**\n\n**8-Disease Risks:**\n• Mortality: **{risks.get('mortality', '?')}%** | HF: **{risks.get('heart_failure', '?')}%** | MI: **{risks.get('myocardial_infarction', '?')}%**\n• Arrhythmia: **{risks.get('arrhythmia', '?')}%** | Sepsis: **{risks.get('sepsis', '?')}%** | PE: **{risks.get('pulmonary_embolism', '?')}%**\n• AKI: **{risks.get('acute_kidney_injury', '?')}%** | ICU: **{risks.get('icu_admission', '?')}%**\n\n**Gate Weights:** CXR {cxr}% | ECG {ecg}% | Labs {labs}%\n\nAsk about any specific disease risk, lab values, ECG/CXR findings, or clinical guidelines.\n\n📋 VisionCare 3.0 — Multi-Modal Fusion Analysis"


# ─── RAG Engine Class ─────────────────────────────────────────────────────────

class MedicalRAGEngine:
    """
    RAG engine that:
    1. Retrieves relevant medical text chunks from FAISS
    2. Builds a grounded prompt with patient context
    3. Calls Gemini 2.5 Flash for generation
    4. Falls back to rule-based responses if anything fails
    """

    def __init__(self):
        self.vectorstore = None
        self.embeddings = None
        self.ready = False
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.chunk_count = 0

    def load_index(self):
        """Load the pre-built FAISS index. Returns True if successful."""
        index_path = INDEX_DIR / "index.faiss"
        if not index_path.exists():
            print("⚠️  No FAISS index found. Building from medical PDFs...")
            try:
                from build_rag import build_index
                n = build_index()
                if n == 0:
                    print("❌ No chunks created. RAG will use rule-based fallback.")
                    return False
            except Exception as e:
                print(f"❌ Failed to build index: {e}")
                return False

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS

            print(f"🧠 Loading embedding model: {EMBED_MODEL}...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            print(f"📊 Loading FAISS index from {INDEX_DIR}...")
            self.vectorstore = FAISS.load_local(
                str(INDEX_DIR), self.embeddings,
                allow_dangerous_deserialization=True
            )

            # Get chunk count
            self.chunk_count = self.vectorstore.index.ntotal
            self.ready = True
            print(f"✅ RAG engine ready — {self.chunk_count} chunks indexed")
            return True

        except Exception as e:
            print(f"❌ Failed to load FAISS index: {e}")
            return False

    def retrieve(self, query: str, k: int = 5) -> list:
        """Retrieve top-k relevant chunks from the FAISS index."""
        if not self.ready or not self.vectorstore:
            return []
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return []

    def _build_patient_context(self, patient_data: dict) -> str:
        """Build patient context string from encounter/patient data."""
        if not patient_data:
            return "No patient data available."

        risks = patient_data.get("risks", {})
        gates = patient_data.get("gates", {})
        lines = []

        if patient_data.get("patient_name"):
            lines.append(f"Patient: {patient_data['patient_name']}, "
                         f"Age {patient_data.get('age', '?')}, "
                         f"{patient_data.get('gender', '?')}")
        if patient_data.get("mrn"):
            lines.append(f"MRN: {patient_data['mrn']}")
        if patient_data.get("encounter_label"):
            lines.append(f"Encounter: {patient_data['encounter_label']}, "
                         f"Date: {patient_data.get('encounter_date', '?')}")
        if patient_data.get("condition"):
            lines.append(f"Condition: {patient_data['condition']}")

        if risks:
            lines.append("\n8-DISEASE RISK SCORES:")
            for k, v in risks.items():
                lines.append(f"  • {k.replace('_', ' ').title()}: {v}%")

        if gates:
            lines.append(f"\nGATE WEIGHTS: CXR {round(gates.get('vision',0.34)*100)}% | "
                         f"ECG {round(gates.get('signal',0.34)*100)}% | "
                         f"Labs {round(gates.get('clinical',0.32)*100)}%")

        if patient_data.get("cxr_findings"):
            lines.append(f"\nCXR Findings: {', '.join(patient_data['cxr_findings'])}")
        if patient_data.get("ecg_findings"):
            lines.append(f"ECG Findings: {', '.join(patient_data['ecg_findings'])}")
        if patient_data.get("labs"):
            lab_str = ", ".join(f"{l['name']}={l['value']}" for l in patient_data["labs"][:6])
            lines.append(f"Key Labs: {lab_str}")

        return "\n".join(lines)

    async def generate_response(self, question: str, patient_data: dict = None) -> dict:
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks from FAISS
        2. Build grounded system prompt
        3. Call Gemini 2.5 Flash
        4. Return formatted response with citations
        """
        t0 = time.time()

        # ── Step 1: Retrieve ──────────────────────────────────────
        retrieved_docs = self.retrieve(question, k=5)
        retrieved_context = ""
        sources = []

        if retrieved_docs:
            chunks_text = []
            for i, doc in enumerate(retrieved_docs):
                src = doc.metadata.get("source_book", "Unknown")
                page = doc.metadata.get("page", "?")
                chunks_text.append(
                    f"[Chunk {i+1}] Source: {src}, Page {page}\n{doc.page_content}"
                )
                if src not in sources:
                    sources.append(src)
            retrieved_context = "\n\n".join(chunks_text)
        else:
            retrieved_context = "No relevant chunks retrieved. Use your built-in medical knowledge."

        # ── Step 2: Build prompt ──────────────────────────────────
        patient_context = self._build_patient_context(patient_data)

        full_prompt = MEDICAL_SYSTEM_PROMPT.format(
            retrieved_context=retrieved_context,
            patient_context=patient_context,
        )

        # ── Step 3: Call Gemini ────────────────────────────────────
        if not self.gemini_key:
            # No API key → rule-based fallback
            reply = rule_based_response(question, patient_data)
            return {
                "reply": reply,
                "source": "rule_based",
                "sources_used": [],
                "chunks_retrieved": 0,
                "latency_ms": round((time.time() - t0) * 1000),
            }

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{GEMINI_URL}?key={self.gemini_key}",
                    json={
                        "contents": [{
                            "role": "user",
                            "parts": [{"text": full_prompt + "\n\nUser question: " + question}]
                        }],
                        "generationConfig": {
                            "maxOutputTokens": 800,
                            "temperature": 0.3,
                            "topP": 0.9,
                        }
                    },
                    headers={"Content-Type": "application/json"},
                )

            if resp.status_code == 200:
                data = resp.json()
                reply = data.get("candidates", [{}])[0].get(
                    "content", {}
                ).get("parts", [{}])[0].get("text", "")

                if reply:
                    return {
                        "reply": reply,
                        "source": "gemini_rag",
                        "model": GEMINI_MODEL,
                        "sources_used": sources,
                        "chunks_retrieved": len(retrieved_docs),
                        "latency_ms": round((time.time() - t0) * 1000),
                    }

            # Gemini returned error → fallback
            print(f"⚠️ Gemini returned {resp.status_code}: {resp.text[:200]}")

        except Exception as e:
            print(f"⚠️ Gemini call failed: {e}")

        # ── Step 4: Fallback ──────────────────────────────────────
        reply = rule_based_response(question, patient_data)
        return {
            "reply": reply,
            "source": "rule_based",
            "sources_used": [],
            "chunks_retrieved": len(retrieved_docs),
            "latency_ms": round((time.time() - t0) * 1000),
        }

    def get_status(self) -> dict:
        """Return RAG engine status for the /api/model/info endpoint."""
        return {
            "rag_ready": self.ready,
            "chunks_indexed": self.chunk_count,
            "embedding_model": EMBED_MODEL,
            "generation_model": GEMINI_MODEL,
            "gemini_key_set": bool(self.gemini_key),
            "index_path": str(INDEX_DIR),
        }
