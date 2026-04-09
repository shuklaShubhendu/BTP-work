// src/pages/About.jsx
import { Heart } from 'lucide-react';

export default function About() {
  return (
    <div className="page-content section-gap">
      <div className="about-hero">
        <div style={{ width:68, height:68, background:'var(--green-dim)', border:'1px solid var(--green-border)', borderRadius:16, display:'flex', alignItems:'center', justifyContent:'center' }}>
          <Heart size={30} color="var(--green)" />
        </div>
        <h1>VisionCare 2.0</h1>
        <p className="about-desc">A multi-modal clinical decision support system combining Chest X-Ray, ECG, and Clinical Lab data for Heart Failure and Mortality prediction using deep learning fusion.</p>
        <div className="tags-row">
          <span className="tag tag-green">Macro AUC: 0.8105</span>
          <span className="tag tag-blue">SYMILE-MIMIC Dataset</span>
          <span className="tag tag-purple">BTP — IIIT Kottayam</span>
          <span className="tag tag-amber">Gemini AI Powered</span>
        </div>
      </div>

      <div className="about-grid">
        <div className="about-card">
          <h3>🧠 Model Architecture (V2)</h3>
          <ul>
            <li><strong>CXR Encoder:</strong> ConvNeXt-Tiny → 768-D embedding</li>
            <li><strong>ECG Encoder:</strong> Custom 1D-CNN → 256-D embedding</li>
            <li><strong>Labs Encoder:</strong> 2-layer MLP → 64-D embedding</li>
            <li><strong>Fusion:</strong> Cross-Attention Gated Fusion</li>
            <li><strong>Output:</strong> Mortality + Heart Failure (multi-label)</li>
            <li><strong>Loss:</strong> Focal Loss (γ=2.0) + Gate Entropy Reg</li>
          </ul>
        </div>
        <div className="about-card">
          <h3>📊 Performance (Val Set)</h3>
          <ul>
            <li><strong>Macro AUC:</strong> <span style={{color:'var(--green)'}}>0.8105</span></li>
            <li><strong>Mortality AUC:</strong> 0.8022</li>
            <li><strong>Heart Failure AUC:</strong> 0.8189</li>
            <li><strong>Training Phase A:</strong> 5 epochs (frozen encoders)</li>
            <li><strong>Training Phase B:</strong> 20 epochs (unfrozen last blocks)</li>
            <li><strong>Best EMA checkpoint:</strong> B-20 epoch</li>
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
            <li><strong>AI Chat:</strong> Gemini + AHA/ESC guideline RAG</li>
            <li><strong>Per-class AUC:</strong> Separate HF and Mortality reporting</li>
            <li><strong>Calibration:</strong> Platt scaling applied</li>
            <li><strong>MAGGIC score:</strong> Validated against external score</li>
          </ul>
        </div>
        <div className="about-card">
          <h3>🔄 V2 vs V3 Comparison</h3>
          <ul>
            <li><strong>V2 CXR:</strong> ConvNeXt-Tiny (ImageNet pre-trained)</li>
            <li><strong>V3 CXR:</strong> ResNet-50 (SYMILE pre-trained) ↑</li>
            <li><strong>V2 ECG:</strong> Custom 1D-CNN — HF: 0.8189</li>
            <li><strong>V3 ECG:</strong> 1D-ResNet18 — HF: 0.7519 ↓</li>
            <li><strong>Gate Reg:</strong> V2=0.01, V3=0.01 (tuned)</li>
            <li><strong>Decision:</strong> V2 is current production baseline</li>
          </ul>
        </div>
        <div className="about-card">
          <h3>⚙️ Technology Stack</h3>
          <ul>
            <li><strong>Frontend:</strong> React 18 + Vite + Recharts</li>
            <li><strong>Backend:</strong> FastAPI + SQLite + SQLAlchemy</li>
            <li><strong>AI Chat:</strong> Google Gemini 1.5 Flash</li>
            <li><strong>ML:</strong> PyTorch 2.10 + CUDA (Tesla T4)</li>
            <li><strong>Pre-training:</strong> SYMILE contrastive learning</li>
            <li><strong>Deployment:</strong> Uvicorn + localhost</li>
          </ul>
        </div>
      </div>

      {/* Team */}
      <div className="card" style={{ textAlign:'center' }}>
        <div className="card-title" style={{ marginBottom:4 }}>Research Team</div>
        <div style={{ fontSize:13, color:'var(--text-secondary)', marginBottom:16 }}>B.Tech Computer Science & Engineering, Semester 7 — IIIT Kottayam</div>
        <div style={{ display:'flex', justifyContent:'center', gap:24 }}>
          {['Arjun Mehta','Priya Krishnan','Rahul Nair'].map(name => (
            <div key={name} style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:8 }}>
              <div style={{ width:48, height:48, borderRadius:'50%', background:'var(--green-dim)', border:'1px solid var(--green-border)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:16, fontWeight:700, color:'var(--green)' }}>
                {name[0]}
              </div>
              <div style={{ fontSize:12, fontWeight:600 }}>{name}</div>
              <div style={{ fontSize:11, color:'var(--text-muted)' }}>BTP Researcher</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
