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

import os, json, time, asyncio
from pathlib import Path
from typing import Optional

INDEX_DIR = Path(__file__).parent / "rag_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT_SEC = float(os.getenv("GEMINI_TIMEOUT_SEC", "35"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

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

5. Keep responses between 140-260 words.

6. Structure responses with clear sections and compact bullet points.

7. Always relate your answer back to the patient's specific data when available.

8. When discussing risk scores, explain what drives the prediction using ALL three modalities (CXR, ECG, Labs).

9. Never claim to diagnose. Use language like "suggests", "consistent with", "warrants further evaluation".

10. FOLLOW THIS OUTPUT TEMPLATE EXACTLY unless user asks for a different format:
### Patient Snapshot
• Name/age/sex + encounter context in one line
• Highest 2 risks with percentages and severity labels (Critical/Moderate/Low)

### 8-Disease Risk Overview
• Mortality, Heart Failure, MI, Arrhythmia, Sepsis, PE, AKI, ICU with percentages
• Keep this as compact grouped bullets (not one long paragraph)

### Clinical Drivers (CXR + ECG + Labs)
• CXR: 1 short finding tied to risk
• ECG: 1 short finding tied to risk
• Labs: 1 short finding tied to risk

### Recommended Clinical Focus
• 2-3 concise next-focus items for physician review
• Include uncertainty language ("suggests", "warrants", "consider")

### Evidence
📋 At least one citation line
"""


# ─── Rule-Based Fallback ──────────────────────────────────────────────────────

def rule_based_response(msg, patient_context):
    """Offline fallback when Gemini is unavailable. Uses patient-specific data, not static text."""
    ml = (msg or "").lower().strip()
    ctx = patient_context or {}
    risks = ctx.get("risks", {}) or {}
    gates = ctx.get("gates", {"vision": 0.339, "signal": 0.338, "clinical": 0.323}) or {}
    cxr = round((gates.get("vision", 0.339)) * 100)
    ecg = round((gates.get("signal", 0.338)) * 100)
    labs = round((gates.get("clinical", 0.323)) * 100)
    name = ctx.get("patient_name", "this patient")

    refusal = (
        "I am VisionCare's Medical AI Assistant. I can only discuss clinical topics related to this "
        "patient's cardiovascular risk assessment. Please ask me about the patient's risk scores, lab values, "
        "ECG/CXR findings, or relevant clinical guidelines."
    )

    off_topic_tokens = [
        "ipl", "cricket", "football", "weather", "recipe", "joke", "politics",
        "movie", "song", "stock", "bitcoin", "code", "python", "java",
    ]
    if any(token in ml for token in off_topic_tokens):
        return refusal

    risk_meta = [
        ("mortality", "Mortality"),
        ("heart_failure", "Heart Failure"),
        ("myocardial_infarction", "Myocardial Infarction"),
        ("arrhythmia", "Arrhythmia"),
        ("sepsis", "Sepsis"),
        ("pulmonary_embolism", "Pulmonary Embolism"),
        ("acute_kidney_injury", "AKI"),
        ("icu_admission", "ICU Admission"),
    ]
    ranked = sorted(
        [(key, label, float(risks.get(key, 0) or 0)) for key, label in risk_meta],
        key=lambda item: item[2],
        reverse=True,
    )

    def bucket(v):
        if v >= 70:
            return "Critical"
        if v >= 40:
            return "Moderate"
        return "Low"

    if not risks:
        return (
            f"**No stored risk profile found for {name}.**\n\n"
            "Please run **Live V3 Inference** for this encounter first, then ask clinical questions.\n\n"
            "📋 VisionCare 3.0 Backend Status"
        )

    if any(w in ml for w in ["highest", "top", "most", "problem", "worse", "worst", "riskiest"]):
        top = ranked[:3]
        return (
            f"**Highest-concern risks for {name}:**\n\n"
            f"• **{top[0][1]}: {top[0][2]:.1f}%** ({bucket(top[0][2])})\n"
            f"• **{top[1][1]}: {top[1][2]:.1f}%** ({bucket(top[1][2])})\n"
            f"• **{top[2][1]}: {top[2][2]:.1f}%** ({bucket(top[2][2])})\n\n"
            "Clinical priority should focus first on the top-ranked risk and related organ support needs.\n\n"
            "📋 VisionCare 3.0 Encounter Risk Ranking"
        )

    if any(w in ml for w in ["overview", "all 8", "eight", "explain all", "summary"]):
        lines = [f"**VisionCare 3.0 risk overview for {name}:**", ""]
        for key, label, value in ranked:
            lines.append(f"• **{label}: {value:.1f}%** ({bucket(value)})")
        lines.append("")
        lines.append("Highest clinical concern is the top-ranked condition above.")
        lines.append("")
        lines.append("📋 VisionCare 3.0 — Stored Encounter Assessment")
        return "\n".join(lines)

    disease_aliases = {
        "heart_failure": ["heart failure", "hf"],
        "mortality": ["mortality", "death", "prognosis"],
        "myocardial_infarction": ["myocardial infarction", "mi", "heart attack"],
        "arrhythmia": ["arrhythmia", "rhythm", "af", "vt"],
        "sepsis": ["sepsis", "infection shock"],
        "pulmonary_embolism": ["pulmonary embolism", "pe", "embolism"],
        "acute_kidney_injury": ["aki", "kidney"],
        "icu_admission": ["icu", "critical care", "admission"],
    }
    for key, aliases in disease_aliases.items():
        if any(alias in ml for alias in aliases):
            value = float(risks.get(key, 0) or 0)
            label = next((name_ for k, name_ in risk_meta if k == key), key)
            return (
                f"**{label} risk: {value:.1f}% ({bucket(value)})**\n\n"
                "This response is from the backend fallback engine using this encounter's stored risk profile. "
                "For richer explanation, enable Gemini key and re-ask.\n\n"
                "📋 VisionCare 3.0 Encounter Risk Profile"
            )

    if any(w in ml for w in ["gate", "balance", "contribut", "modal"]):
        return (
            "**Modality weighting (for model interpretation):**\n\n"
            f"• **CXR: {cxr}%**\n"
            f"• **ECG: {ecg}%**\n"
            f"• **Labs: {labs}%**\n\n"
            "These are model-side fusion weights, not direct clinical severity scores.\n\n"
            "📋 VisionCare 3.0 Fusion Gate Status"
        )

    return (
        f"**Clinical summary for {name}:**\n\n"
        f"• Highest risk: **{ranked[0][1]} {ranked[0][2]:.1f}%** ({bucket(ranked[0][2])})\n"
        f"• Second highest: **{ranked[1][1]} {ranked[1][2]:.1f}%** ({bucket(ranked[1][2])})\n"
        f"• Third highest: **{ranked[2][1]} {ranked[2][2]:.1f}%** ({bucket(ranked[2][2])})\n\n"
        "Ask me to explain any one disease risk in detail.\n\n"
        "📋 VisionCare 3.0 Encounter Clinical Summary"
    )


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
        self.gemini_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        self.chunk_count = 0
        self.last_gemini_error = None
        self._gemini_model = None

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
        if patient_data.get("analysis_source"):
            lines.append(f"Analysis Source: {patient_data.get('analysis_source')}")
        if patient_data.get("last_inference_at"):
            lines.append(f"Last Inference: {patient_data.get('last_inference_at')}")

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

    def _get_gemini_model(self):
        if self._gemini_model is not None:
            return self._gemini_model
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_key)
        self._gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        return self._gemini_model

    async def _call_gemini(self, full_prompt: str, question: str):
        """
        Call Gemini SDK with retry/backoff for transient overload/availability failures.
        Returns: (reply_text | None, error_message | None, status_code | None)
        """
        prompt = full_prompt + "\n\nUser question: " + question

        attempts = max(1, GEMINI_MAX_RETRIES)
        for attempt in range(1, attempts + 1):
            try:
                model = self._get_gemini_model()
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        model.generate_content,
                        prompt,
                        generation_config={
                            "max_output_tokens": 700,
                            "temperature": 0.25,
                            "top_p": 0.9,
                        },
                    ),
                    timeout=GEMINI_TIMEOUT_SEC,
                )

                reply = getattr(response, "text", None)
                if reply:
                    return reply, None, 200

                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    try:
                        parts = candidates[0].content.parts
                        text = "".join(getattr(p, "text", "") for p in parts).strip()
                        if text:
                            return text, None, 200
                    except Exception:
                        pass

                return None, "Gemini SDK returned empty response", 200

            except Exception as exc:
                message = f"Gemini SDK call failed: {exc}"
                lowered = message.lower()
                is_retryable = (
                    "503" in lowered or "unavailable" in lowered or "resourceexhausted" in lowered
                    or "429" in lowered or "deadline" in lowered or "timeout" in lowered
                )
                if is_retryable and attempt < attempts:
                    await asyncio.sleep(min(1.2 * attempt, 3.5))
                    continue
                return None, message, None

        return None, "Gemini call exhausted retries", None

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

        reply, err_msg, status_code = await self._call_gemini(full_prompt, question)
        if reply:
            self.last_gemini_error = None
            return {
                "reply": reply,
                "source": "gemini_rag",
                "model": GEMINI_MODEL,
                "sources_used": sources,
                "chunks_retrieved": len(retrieved_docs),
                "latency_ms": round((time.time() - t0) * 1000),
            }
        self.last_gemini_error = err_msg
        if err_msg:
            print(f"⚠️ {err_msg}")

        # ── Step 4: Fallback ──────────────────────────────────────
        reply = rule_based_response(question, patient_data)
        return {
            "reply": reply,
            "source": "rule_based",
            "sources_used": [],
            "chunks_retrieved": len(retrieved_docs),
            "fallback_reason": self.last_gemini_error or "gemini_unavailable",
            "upstream_status": status_code,
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
            "gemini_timeout_sec": GEMINI_TIMEOUT_SEC,
            "gemini_max_retries": GEMINI_MAX_RETRIES,
            "last_gemini_error": self.last_gemini_error,
            "index_path": str(INDEX_DIR),
        }
