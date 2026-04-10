// src/pages/About.jsx
import { Heart } from 'lucide-react';

export default function About() {
  return (
    <div className="page-content section-gap">
      <div className="about-hero">
        <div style={{ width:68, height:68, background:'var(--green-dim)', border:'1px solid var(--green-border)', borderRadius:16, display:'flex', alignItems:'center', justifyContent:'center' }}>
          <Heart size={30} color="var(--green)" />
        </div>
        <h1>VisionCare 3.0</h1>
        <p className="about-desc">A multi-modal clinical decision support system combining Chest X-Ray, ECG, and Clinical Lab data for prediction of 8 critical diseases using progressive unfreeze deep learning fusion with balanced modality gates.</p>
        <div className="tags-row">
          <span className="tag tag-green">Macro AUC: 0.7926</span>
          <span className="tag tag-blue">8 Disease Targets</span>
          <span className="tag tag-purple">BTP — IIIT Kottayam</span>
          <span className="tag tag-amber">Balanced Gates: 34/34/32</span>
        </div>
      </div>

      <div className="about-grid">
        <div className="about-card">
          <h3>Model Architecture (V3)</h3>
          <ul>
            <li><strong>CXR Encoder:</strong> ResNet-50 (SYMILE) → 2048-D embedding</li>
            <li><strong>ECG Encoder:</strong> 1D ResNet-18 (SYMILE) → 512-D embedding</li>
            <li><strong>Labs Encoder:</strong> 3-layer NN (SYMILE) → 256-D embedding</li>
            <li><strong>Fusion:</strong> Cross-Attention (4-head) + Gate Entropy Reg</li>
            <li><strong>Output:</strong> 8 diseases (Mort, HF, MI, Arr, Sep, PE, AKI, ICU)</li>
            <li><strong>Training:</strong> Progressive Unfreeze (Phase A+B), Focal Loss, EMA</li>
          </ul>
        </div>
        <div className="about-card">
          <h3>Performance (Val Set)</h3>
          <ul>
            <li><strong>Macro AUC:</strong> <span style={{color:'var(--green)'}}>0.7926</span> (8 diseases)</li>
            <li><strong>Mortality AUC:</strong> 0.8082</li>
            <li><strong>Sepsis AUC:</strong> 0.8223 (highest)</li>
            <li><strong>Heart Failure AUC:</strong> 0.7829</li>
            <li><strong>Gate Balance:</strong> V=33.9% S=33.8% C=32.3%</li>
            <li><strong>Best EMA checkpoint:</strong> Epoch 17/25</li>
          </ul>
        </div>
        <div className="about-card">
          <h3>📋 Dataset — SYMILE-MIMIC</h3>
          <ul>
            <li><strong>Source:</strong> PhysioNet / SYMILE (MIT)</li>
            <li><strong>Total Admissions:</strong> 11,622</li>
            <li><strong>Training Set:</strong> 10,000 admissions</li>
            <li><strong>Validation Set:</strong> 750 admissions</li>
            <li><strong>Mortality Labels:</strong> Real (hospital_expire_flag)</li>
            <li><strong>HF Labels:</strong> Clinical proxy (Edema+Cardiomegaly+Effusion)</li>
          </ul>
        </div>
        <div className="about-card">
          <h3>🔬 XAI & Explainability</h3>
          <ul>
            <li><strong>Grad-CAM:</strong> CXR region saliency maps</li>
            <li><strong>Modality Gates:</strong> Per-modality contribution weights</li>
            <li><strong>AI Chat:</strong> RAG (FAISS + Medical PDFs) + Gemini 2.5 Flash (backend)</li>
            <li><strong>Per-class AUC:</strong> Separate HF and Mortality reporting</li>
            <li><strong>Calibration:</strong> Platt scaling applied</li>
            <li><strong>MAGGIC score:</strong> Validated against external score</li>
          </ul>
        </div>
        <div className="about-card">
          <h3>V2 Phase 2 vs V3 Comparison</h3>
          <ul>
            <li><strong>V2P2 CXR:</strong> ConvNeXt-Tiny (frozen) → Gates: 79% Vision</li>
            <li><strong>V3 CXR:</strong> ResNet-50 (progressive unfreeze) → 34% Vision</li>
            <li><strong>V2P2 AUC:</strong> 0.7672 (imbalanced gates)</li>
            <li><strong>V3 AUC:</strong> 0.7926 (+3.3% improvement)</li>
            <li><strong>Key Diff:</strong> Progressive Unfreeze + Gate Entropy Reg</li>
            <li><strong>Result:</strong> V3 is production model (balanced 34/34/32)</li>
          </ul>
        </div>
        <div className="about-card">
          <h3>⚙️ Technology Stack</h3>
          <ul>
            <li><strong>Frontend:</strong> React 18 + Vite + Recharts</li>
            <li><strong>Backend:</strong> FastAPI + PyTorch (CPU inference)</li>
            <li><strong>AI Chat:</strong> RAG (FAISS + LangChain) + Gemini 2.5 Flash</li>
            <li><strong>ML:</strong> PyTorch 2.1 + CUDA (Tesla T4 for training)</li>
            <li><strong>Pre-training:</strong> SYMILE contrastive learning</li>
            <li><strong>Deployment:</strong> Docker Compose (frontend + backend)</li>
          </ul>
        </div>
      </div>

      {/* Team */}
      <div className="card" style={{ textAlign:'center' }}>
        <div className="card-title" style={{ marginBottom:4 }}>Research Team</div>
        <div style={{ fontSize:13, color:'var(--text-secondary)', marginBottom:16 }}>B.Tech Computer Science & Engineering, Semester 7 — IIIT Kottayam</div>
        <div style={{ display:'flex', flexWrap:'wrap', justifyContent:'center', gap:20, marginBottom:18 }}>
          {[
            { name: 'Shubhendu Shukla', roll: '2022BCD0054' },
            { name: 'Aditi Saugat', roll: '2022BCS0200' },
            { name: 'Duggammagari Jayanth Kumar Reddy', roll: '2022BCD0042' },
            { name: 'B Akash', roll: '2022BCS0087' },
          ].map((member) => (
            <div key={member.roll} style={{ width:220, display:'flex', flexDirection:'column', alignItems:'center', gap:8, padding:'10px 12px', borderRadius:12, border:'1px solid var(--border)', background:'var(--bg-elevated)' }}>
              <div style={{ width:56, height:56, borderRadius:'50%', background:'var(--green-dim)', border:'1px solid var(--green-border)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:18, fontWeight:700, color:'var(--green)' }}>
                {member.name.split(' ').map((x) => x[0]).join('').slice(0, 2)}
              </div>
              <div style={{ fontSize:13, fontWeight:700, lineHeight:1.35 }}>{member.name}</div>
              <div style={{ fontSize:12, color:'var(--text-secondary)' }}>{member.roll}</div>
              <div style={{ fontSize:11, color:'var(--text-muted)' }}>BTP Researcher</div>
            </div>
          ))}
        </div>
        <div style={{ margin:'0 auto', maxWidth:360, border:'1px solid var(--green-border)', background:'var(--green-dim)', borderRadius:12, padding:'12px 14px' }}>
          <div style={{ fontSize:12, color:'var(--text-secondary)', textTransform:'uppercase', letterSpacing:'0.08em' }}>Guided By</div>
          <div style={{ fontSize:24, fontWeight:700, marginTop:4 }}>Dr. Dhakshayani J</div>
        </div>
      </div>
    </div>
  );
}
