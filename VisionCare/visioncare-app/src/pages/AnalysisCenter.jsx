// src/pages/AnalysisCenter.jsx — VisionCare 3.0 Single & Multi-modal Analysis
import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, CheckCircle, ChevronRight, Zap, RotateCcw, UserPlus } from 'lucide-react';
import RiskGauge from '../components/RiskGauge';
import ECGViewer from '../components/ECGViewer';
import CXRViewer from '../components/CXRViewer';
import { V3_TARGETS } from '../utils/mockData';
import { analyze as analyzeAPI, analyzeCxr, analyzeEcg, analyzeLabs } from '../utils/api';

const DEFAULT_LABS = { bnp:'', troponin:'', creatinine:'', sodium:'', potassium:'', hemoglobin:'', wbc:'', glucose:'' };

const LAB_CONFIG = [
  { key:'bnp',        label:'BNP',        unit:'pg/mL',   normal:'<100',     placeholder:'e.g. 850' },
  { key:'troponin',   label:'Troponin I', unit:'ng/mL',   normal:'<0.04',    placeholder:'e.g. 0.04' },
  { key:'creatinine', label:'Creatinine', unit:'mg/dL',   normal:'0.7-1.3',  placeholder:'e.g. 1.8' },
  { key:'sodium',     label:'Sodium',     unit:'mEq/L',   normal:'136-145',  placeholder:'e.g. 138' },
  { key:'potassium',  label:'Potassium',  unit:'mEq/L',   normal:'3.5-5.0',  placeholder:'e.g. 4.2' },
  { key:'hemoglobin', label:'Hemoglobin', unit:'g/dL',    normal:'12-17.5',  placeholder:'e.g. 11.2' },
  { key:'wbc',        label:'WBC',        unit:'×10³/μL', normal:'4.5-11.0', placeholder:'e.g. 9.8' },
  { key:'glucose',    label:'Glucose',    unit:'mg/dL',   normal:'70-100',   placeholder:'e.g. 142' },
];

function scoreLabs(labs) {
  let score = 0;
  if (parseFloat(labs.bnp)        > 400)  score += 35;
  if (parseFloat(labs.troponin)   > 0.04) score += 20;
  if (parseFloat(labs.creatinine) > 1.5)  score += 15;
  if (parseFloat(labs.sodium)     < 135)  score += 12;
  if (parseFloat(labs.hemoglobin) < 12)   score += 8;
  if (parseFloat(labs.glucose)    > 120)  score += 5;
  return Math.min(score, 95);
}

function MockPredict(mode, labs, hasCxr, hasEcg) {
  const labScore = scoreLabs(labs);
  const cxrBonus = hasCxr ? 12 : 0;
  const ecgBonus = hasEcg ? 10 : 0;
  const base = labScore * 0.6 + cxrBonus + ecgBonus;
  const clamp = (v) => Math.min(Math.max(Math.round(v), 2), 95);
  const risks = {
    mortality: clamp(base * 0.38 + Math.random() * 8),
    heart_failure: clamp(base * 1.0 + Math.random() * 12),
    myocardial_infarction: clamp(base * 0.25 + Math.random() * 6),
    arrhythmia: clamp(base * 0.45 + Math.random() * 10),
    sepsis: clamp(base * 0.30 + Math.random() * 8),
    pulmonary_embolism: clamp(base * 0.12 + Math.random() * 4),
    acute_kidney_injury: clamp(base * 0.55 + Math.random() * 10),
    icu_admission: clamp(base * 0.50 + Math.random() * 10),
  };
  const labW = mode === 'labs'  ? 0.90 : mode === 'multimodal' ? 0.34 : 0.0;
  const cxrW = mode === 'cxr'   ? 0.85 : mode === 'multimodal' ? 0.34 : 0.0;
  const ecgW = mode === 'ecg'   ? 0.80 : mode === 'multimodal' ? 0.32 : 0.0;
  const total = labW + cxrW + ecgW || 1;
  return { risks, gates:{ vision: cxrW/total, signal: ecgW/total, clinical: labW/total } };
}

export default function AnalysisCenter() {
  const navigate = useNavigate();
  const [mode, setMode]       = useState('multimodal'); // multimodal | cxr | ecg | labs
  const [step, setStep]       = useState(0);
  const [labs, setLabs]       = useState(DEFAULT_LABS);
  const [cxrFile, setCxrFile] = useState(null);
  const [cxrUrl, setCxrUrl]   = useState(null);
  const [hasEcg, setHasEcg]   = useState(false);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef();
  const cancelledRef = useRef(false);

  const MODE_STEPS = {
    multimodal: ['Upload CXR', 'ECG Signal', 'Lab Values', 'Prediction'],
    cxr:        ['Upload CXR', 'Prediction'],
    ecg:        ['ECG Signal', 'Prediction'],
    labs:       ['Lab Values', 'Prediction'],
  };
  const steps = MODE_STEPS[mode];

  const handleCxrUpload = (file) => {
    setCxrFile(file);
    setCxrUrl(URL.createObjectURL(file));
    setStep(s => Math.min(s + 1, steps.length - 2));
  };

  const handleAnalyze = async () => {
    cancelledRef.current = false;
    setLoading(true);

    const cleanLabs = Object.fromEntries(
      Object.entries(labs)
        .filter(([, v]) => v !== '')
        .map(([k, v]) => [k, parseFloat(v)])
    );

    let res;
    try {
      if (mode === 'cxr') {
        res = await analyzeCxr(cxrFile);
      } else if (mode === 'ecg') {
        res = await analyzeEcg();
      } else if (mode === 'labs') {
        res = await analyzeLabs(cleanLabs);
      } else {
        // multimodal: fusion model (labs + has_ecg flags; CXR sent separately if needed)
        res = await analyzeAPI({ labs: cleanLabs, has_ecg: hasEcg, mode });
      }
    } catch (err) {
      console.warn('Backend unavailable — falling back to mock:', err.message);
      res = MockPredict(mode, labs, !!cxrFile, hasEcg || mode === 'ecg');
    }

    if (cancelledRef.current) return;
    setResult(res);
    setStep(steps.length - 1);
    setLoading(false);
  };

  const reset = () => {
    cancelledRef.current = true;
    setStep(0); setLabs(DEFAULT_LABS); setCxrFile(null); setCxrUrl(null);
    setHasEcg(false); setResult(null); setLoading(false);
  };

  const canAnalyze = (mode === 'cxr' && cxrFile) || (mode === 'ecg' && hasEcg) ||
    (mode === 'labs' && labs.bnp) || (mode === 'multimodal' && (cxrFile || labs.bnp || hasEcg));


  return (
    <div className="page-content section-gap">
      {/* Onboard CTA */}
      <div className="card" onClick={() => navigate('/onboard')}
        style={{ padding:'16px 20px', cursor:'pointer', background:'linear-gradient(135deg, rgba(34,197,94,0.06), rgba(34,197,94,0.02))', border:'1px solid var(--green-border)', transition:'all 0.2s' }}
        onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--green)'}
        onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--green-border)'}>
        <div style={{ display:'flex', alignItems:'center', gap:14 }}>
          <div style={{ width:40, height:40, borderRadius:10, background:'var(--green-dim)', display:'flex', alignItems:'center', justifyContent:'center' }}>
            <UserPlus size={20} color="var(--green)" />
          </div>
          <div style={{ flex:1 }}>
            <div style={{ fontWeight:700, fontSize:14 }}>🏥 Onboard New Patient</div>
            <div style={{ fontSize:12, color:'var(--text-secondary)', marginTop:2 }}>
              Register patient details → upload clinical data → get instant V3 risk assessment
            </div>
          </div>
          <div style={{ color:'var(--green)', fontSize:12, fontWeight:600, display:'flex', alignItems:'center', gap:4 }}>
            Start <span style={{ fontSize:16 }}>→</span>
          </div>
        </div>
      </div>

      {/* Mode selector */}
      <div className="card" style={{ padding:'14px 18px' }}>
        <div style={{ fontSize:13, fontWeight:600, marginBottom:12, color:'var(--text-secondary)' }}>SELECT ANALYSIS MODE</div>
        <div style={{ display:'flex', gap:10 }}>
          {[['multimodal','🔀 Multi-Modal (Fusion)'],['cxr','🫁 CXR Only'],['ecg','💓 ECG Only'],['labs','🧪 Labs Only']].map(([m, label]) => (
            <button key={m} onClick={() => { setMode(m); reset(); }}
              style={{ flex:1, padding:'10px', background: mode===m ? 'var(--green-dim)' : 'var(--bg-input)', border:`1px solid ${mode===m?'var(--green-border)':'var(--border)'}`, borderRadius:8, color: mode===m?'var(--green)':'var(--text-secondary)', fontSize:12, fontWeight: mode===m?700:400, cursor:'pointer', transition:'all 0.15s' }}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Step indicators */}
      <div className="step-indicators">
        {steps.map((label, i) => (
          <div key={i} className="step-item">
            <div className={`step-circle ${i < step ? 'done' : i === step ? 'active' : ''}`}>
              {i < step ? '✓' : i + 1}
            </div>
            <div className={`step-label ${i === step ? 'active' : ''}`}>{label}</div>
          </div>
        ))}
      </div>

      <div className="analysis-layout">
        {/* LEFT: Input area */}
        <div className="section-gap">

          {/* CXR Upload */}
          {(mode === 'multimodal' || mode === 'cxr') && (
            <div className="card">
              <div className="card-title">🫁 Chest X-Ray Upload</div>
              {cxrUrl ? (
                <div>
                  <div style={{ color:'var(--green)', fontSize:13, fontWeight:600, marginBottom:4 }}><CheckCircle size={14} style={{ marginRight:4 }} />CXR uploaded successfully</div>
                  <div style={{ fontSize:12, color:'var(--text-secondary)', marginBottom:10 }}>{cxrFile?.name}</div>
                  <CXRViewer imageSrc={cxrUrl} findings={['Uploaded CXR']} severity="moderate" />
                  <button onClick={() => { setCxrFile(null); setCxrUrl(null); }} style={{ marginTop:10, padding:'5px 12px', background:'none', border:'1px solid var(--border)', borderRadius:6, color:'var(--text-secondary)', fontSize:12, cursor:'pointer' }}>Remove</button>
                </div>
              ) : (
                <div className="upload-zone" onClick={() => fileRef.current.click()}
                  onDragOver={e => { e.preventDefault(); e.currentTarget.classList.add('drag-over'); }}
                  onDragLeave={e => e.currentTarget.classList.remove('drag-over')}
                  onDrop={e => { e.preventDefault(); e.currentTarget.classList.remove('drag-over'); handleCxrUpload(e.dataTransfer.files[0]); }}>
                  <Upload size={32} color="var(--text-muted)" style={{ marginBottom:8 }} />
                  <h4>Drop CXR image here or click to upload</h4>
                  <p>Supports JPG, PNG · Max 10MB</p>
                  <p style={{ marginTop:4, fontSize:11, color:'var(--text-muted)' }}>Use NIH ChestX-ray14 or MIMIC-CXR images for best results</p>
                  <input ref={fileRef} type="file" accept="image/*" style={{ display:'none' }} onChange={e => e.target.files[0] && handleCxrUpload(e.target.files[0])} />
                </div>
              )}
            </div>
          )}

          {/* ECG */}
          {(mode === 'multimodal' || mode === 'ecg') && (
            <div className="card">
              <div className="card-title">💓 ECG Signal</div>
              {hasEcg ? (
                <div>
                  <div style={{ color:'var(--green)', fontSize:13, fontWeight:600, marginBottom:8 }}><CheckCircle size={14} style={{ marginRight:4 }} />ECG loaded — demo 12-lead signal</div>
                  <ECGViewer findings={['Atrial fibrillation detected','ST-segment elevation in V2-V4','Left ventricular hypertrophy criteria met']} />
                  <button onClick={() => setHasEcg(false)} style={{ marginTop:8, padding:'5px 12px', background:'none', border:'1px solid var(--border)', borderRadius:6, color:'var(--text-secondary)', fontSize:12, cursor:'pointer' }}>Remove</button>
                </div>
              ) : (
                <div style={{ display:'flex', gap:10 }}>
                  <button onClick={() => setHasEcg(true)} style={{ flex:1, padding:12, background:'var(--bg-input)', border:'1px solid var(--border)', borderRadius:8, color:'var(--text-primary)', fontSize:13, cursor:'pointer', transition:'all 0.15s' }}>
                    📂 Load Demo ECG (.npy)
                  </button>
                  <button style={{ flex:1, padding:12, background:'var(--bg-input)', border:'1px solid var(--border)', borderRadius:8, color:'var(--text-secondary)', fontSize:13, cursor:'pointer' }}>
                    ⬆️ Upload ECG File
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Lab Values */}
          {(mode === 'multimodal' || mode === 'labs') && (
            <div className="card">
              <div className="card-title">🧪 Clinical Lab Values</div>
              <div className="labs-form">
                {LAB_CONFIG.map(({ key, label, unit, normal, placeholder }) => (
                  <div key={key} className="lab-input-group">
                    <label>{label} <span className="unit">({unit})</span></label>
                    <input type="number" step="any" placeholder={placeholder} value={labs[key]}
                      onChange={e => setLabs(l => ({ ...l, [key]: e.target.value }))} />
                    <span className="unit">Normal: {normal}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Analyze button */}
          {step < steps.length - 1 && (
            <button className="analyze-btn" onClick={handleAnalyze} disabled={!canAnalyze || loading}>
              {loading ? '⚙️ Running Inference...' : <><Zap size={16} style={{ marginRight:6 }} />Run {mode === 'multimodal' ? 'Multi-Modal Fusion' : 'Single-Modal'} Analysis</>}
            </button>
          )}
        </div>

        {/* RIGHT: Results */}
        <div className="section-gap">
          {result ? (
            <div className="result-card">
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                <div style={{ fontSize:14, fontWeight:700 }}>Prediction Results</div>
                <button onClick={reset} style={{ background:'none', border:'1px solid var(--border)', borderRadius:6, padding:'5px 10px', color:'var(--text-secondary)', fontSize:11, cursor:'pointer', display:'flex', alignItems:'center', gap:4 }}>
                  <RotateCcw size={11} /> New Analysis
                </button>
              </div>

              {/* 8 Disease Gauges */}
              <div style={{ display:'grid', gridTemplateColumns:'repeat(4, 1fr)', gap:8, marginTop:14 }}>
                {V3_TARGETS.map(t => (
                  <RiskGauge key={t.key} pct={result.risks?.[t.key] || 0} label={t.short} size={90} />
                ))}
              </div>

              {/* Explanation */}
              <div className="result-explanation">
                <strong>Clinical Interpretation (V3 — 8 Diseases):</strong><br/>
                {(() => {
                  const maxKey = Object.entries(result.risks || {}).sort(([,a],[,b]) => b - a)[0]?.[0];
                  const maxVal = result.risks?.[maxKey] || 0;
                  return <span>Highest risk: <strong>{maxKey?.replace(/_/g,' ')} at {maxVal}%</strong>.</span>;
                })()}
                {(result.risks?.heart_failure >= 70 || result.risks?.mortality >= 50) && <><br/><br/>⚠️ <strong style={{color:'var(--text-red)'}}>Critical risk detected.</strong> Immediate cardiology review recommended.</>}
                {(result.risks?.heart_failure >= 40 && result.risks?.heart_failure < 70) && <><br/><br/>🟡 <strong style={{color:'var(--text-amber)'}}>Moderate risk.</strong> Close monitoring advised.</>}
                {(result.risks?.heart_failure < 40 && result.risks?.mortality < 25) && <><br/><br/>🟢 <strong style={{color:'var(--text-green)'}}>Low risk.</strong> Routine follow-up.</>}
                <br/><br/>
                <span style={{ fontSize:11, color: result.real_inference ? 'var(--green)' : 'var(--text-muted)' }}>
                  {result.real_inference ? '✅ Real inference · ' : '⚙️ Mock prediction · '}
                  VisionCare {result.model_version ?? '3.0'} · Macro AUC 0.7926 · 8 targets · Balanced Gates
                </span>
              </div>
            </div>
          ) : (
            <div className="card" style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', minHeight:300, gap:12, color:'var(--text-muted)' }}>
              <Zap size={36} color="var(--text-muted)" />
              <div style={{ fontSize:14, fontWeight:600 }}>Ready to Analyze</div>
              <div style={{ fontSize:12, textAlign:'center', maxWidth:200 }}>
                {mode === 'multimodal' ? 'Upload CXR, load ECG, and enter lab values to run fusion analysis.' : `Provide ${mode.toUpperCase()} data to run single-modal analysis.`}
              </div>
              <div style={{ fontSize:11, color:'var(--text-muted)', background:'var(--bg-elevated)', borderRadius:8, padding:'8px 14px', marginTop:4 }}>
                Mode: <strong style={{color:'var(--green)'}}>{mode === 'multimodal' ? 'Multi-Modal Fusion (V3)' : `${mode.toUpperCase()} Only (Single-Modal)`}</strong>
              </div>
            </div>
          )}

          {/* Info card */}
          <div className="card" style={{ fontSize:12, color:'var(--text-secondary)', lineHeight:1.7 }}>
            <div style={{ fontWeight:700, color:'var(--text-primary)', marginBottom:8 }}>📁 Demo Image Sources</div>
            <div>• <strong>NIH ChestX-ray14</strong>: nihcc.app.box.com/v/ChestXray-NIHCC</div>
            <div style={{ marginTop:4 }}>• <strong>MIMIC-CXR:</strong> physionet.org/content/mimic-cxr</div>
            <div style={{ marginTop:4 }}>• <strong>CheXpert:</strong> stanfordmlgroup.github.io/competitions/chexpert</div>
            <div style={{ marginTop:8, color:'var(--text-muted)', fontSize:11 }}>Download any .jpg and drop it in the CXR upload area above.</div>
          </div>
        </div>
      </div>
    </div>
  );
}
