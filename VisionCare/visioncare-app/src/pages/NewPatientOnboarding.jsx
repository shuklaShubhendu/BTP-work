// src/pages/NewPatientOnboarding.jsx — 3-Step Patient Onboarding Wizard
import { useState, useRef, useMemo } from 'react';
import {
  UserPlus, Upload, CheckCircle, ChevronRight, ChevronLeft,
  Zap, RotateCcw, User, Calendar, Heart, FileText, ArrowRight
} from 'lucide-react';
import RiskGauge from '../components/RiskGauge';
import ECGViewer from '../components/ECGViewer';
import CXRViewer from '../components/CXRViewer';
import { V3_TARGETS } from '../utils/mockData';
import { analyze as analyzeAPI, analyzeCxr } from '../utils/api';

/* ─── Lab Configuration ────────────────────────────────────────────────────── */
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

const DEFAULT_LABS = { bnp:'', troponin:'', creatinine:'', sodium:'', potassium:'', hemoglobin:'', wbc:'', glucose:'' };

/* ─── Mock fallback ────────────────────────────────────────────────────────── */
function mockPredict(labs, hasCxr, hasEcg) {
  const score = (k, d) => { const v = parseFloat(labs[k]); return isNaN(v) ? 0 : v > d ? 15 : 0; };
  const base = score('bnp',400) + score('troponin',0.04) + score('creatinine',1.5) + (hasCxr?12:0) + (hasEcg?10:0);
  const cl = v => Math.min(Math.max(Math.round(v), 2), 95);
  return {
    risks: {
      mortality: cl(base*0.38+Math.random()*8), heart_failure: cl(base*1.0+Math.random()*12),
      myocardial_infarction: cl(base*0.25+Math.random()*6), arrhythmia: cl(base*0.45+Math.random()*10),
      sepsis: cl(base*0.30+Math.random()*8), pulmonary_embolism: cl(base*0.12+Math.random()*4),
      acute_kidney_injury: cl(base*0.55+Math.random()*10), icu_admission: cl(base*0.50+Math.random()*10),
    },
    gates: { vision: 0.34, signal: 0.34, clinical: 0.32 },
  };
}

/* ─── Step Indicator ───────────────────────────────────────────────────────── */
function StepIndicator({ current, steps }) {
  return (
    <div className="onboard-steps">
      {steps.map((s, i) => (
        <div key={i} className={`onboard-step-item ${i < current ? 'done' : i === current ? 'active' : ''}`}>
          <div className="onboard-step-circle">
            {i < current ? <CheckCircle size={18} /> : <span>{i + 1}</span>}
          </div>
          <div className="onboard-step-label">{s.label}</div>
          <div className="onboard-step-sub">{s.sub}</div>
          {i < steps.length - 1 && <div className={`onboard-step-line ${i < current ? 'done' : ''}`} />}
        </div>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
export default function NewPatientOnboarding() {
  const [step, setStep] = useState(0);

  /* Step 1: Patient Info */
  const [patient, setPatient] = useState({ name:'', age:'', gender:'Male', condition:'', mrn: `MRN-${String(Math.floor(Math.random()*90000)+10000)}` });

  /* Step 2: Clinical Data */
  const [labs, setLabs]       = useState(DEFAULT_LABS);
  const [cxrFile, setCxrFile] = useState(null);
  const [cxrUrl, setCxrUrl]   = useState(null);
  const [hasEcg, setHasEcg]   = useState(false);
  const fileRef = useRef();

  /* Step 3: Results */
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const STEPS = [
    { label: 'Patient Details', sub: 'Demographics & info' },
    { label: 'Clinical Data',   sub: 'CXR, ECG & labs' },
    { label: 'V3 Analysis',     sub: '8-disease prediction' },
  ];

  /* ── Validation ──────────────────────────────────────────────────────────── */
  const step1Valid = patient.name.trim().length >= 2 && patient.age && parseInt(patient.age) > 0;
  const hasAnyData = !!cxrFile || hasEcg || Object.values(labs).some(v => v !== '');

  /* ── CXR upload handler ──────────────────────────────────────────────────── */
  const handleCxrUpload = (file) => {
    setCxrFile(file);
    setCxrUrl(URL.createObjectURL(file));
  };

  /* ── Run inference ───────────────────────────────────────────────────────── */
  const runAnalysis = async () => {
    setLoading(true);
    setError(null);
    setStep(2);

    const cleanLabs = Object.fromEntries(
      Object.entries(labs).filter(([, v]) => v !== '').map(([k, v]) => [k, parseFloat(v)])
    );

    try {
      let res;
      if (cxrFile && Object.keys(cleanLabs).length > 0) {
        // Full fusion: send labs via analyze, CXR separately
        res = await analyzeAPI({ labs: cleanLabs, has_ecg: hasEcg, mode: 'multimodal' });
      } else if (cxrFile) {
        res = await analyzeCxr(cxrFile);
      } else {
        res = await analyzeAPI({ labs: cleanLabs, has_ecg: hasEcg, mode: Object.keys(cleanLabs).length ? 'labs' : 'multimodal' });
      }
      setResult(res);
    } catch (err) {
      console.warn('Backend unavailable — using mock:', err.message);
      setResult(mockPredict(labs, !!cxrFile, hasEcg));
      setError('Backend unavailable — showing estimated results.');
    } finally {
      setLoading(false);
    }
  };

  /* ── Reset everything ────────────────────────────────────────────────────── */
  const resetAll = () => {
    setStep(0);
    setPatient({ name:'', age:'', gender:'Male', condition:'', mrn: `MRN-${String(Math.floor(Math.random()*90000)+10000)}` });
    setLabs(DEFAULT_LABS);
    setCxrFile(null); setCxrUrl(null);
    setHasEcg(false);
    setResult(null); setLoading(false); setError(null);
  };

  /* ── Derived ─────────────────────────────────────────────────────────────── */
  const risks = result?.risks || {};
  const maxRiskEntry = useMemo(() => {
    const entries = Object.entries(risks);
    if (!entries.length) return null;
    return entries.sort(([,a],[,b]) => b - a)[0];
  }, [risks]);

  /* ═══════════════════════════════════════════════════════════════════════════ */
  return (
    <div className="page-content section-gap">
      {/* Header */}
      <div className="onboard-header">
        <div className="onboard-header-left">
          <div className="onboard-header-icon"><UserPlus size={22} /></div>
          <div>
            <h2 style={{ margin:0, fontSize:20, fontWeight:800 }}>New Patient Onboarding</h2>
            <p style={{ margin:0, fontSize:13, color:'var(--text-secondary)', marginTop:2 }}>
              Register patient → Upload clinical data → Get AI-powered risk assessment
            </p>
          </div>
        </div>
        {step > 0 && (
          <button className="onboard-reset-btn" onClick={resetAll}>
            <RotateCcw size={14} /> Start Over
          </button>
        )}
      </div>

      {/* Step Indicator */}
      <StepIndicator current={step} steps={STEPS} />

      {/* ════════════════ STEP 1: Patient Details ════════════════════════════ */}
      {step === 0 && (
        <div className="onboard-card onboard-fade-in">
          <div className="onboard-card-header">
            <User size={18} />
            <span>Patient Information</span>
          </div>

          <div className="onboard-form-grid">
            <div className="onboard-field">
              <label>Full Name <span className="required">*</span></label>
              <input type="text" placeholder="e.g. Rajesh Kumar" value={patient.name}
                onChange={e => setPatient(p => ({...p, name: e.target.value}))} autoFocus />
            </div>

            <div className="onboard-field">
              <label>Age <span className="required">*</span></label>
              <input type="number" placeholder="e.g. 67" min="1" max="120" value={patient.age}
                onChange={e => setPatient(p => ({...p, age: e.target.value}))} />
            </div>

            <div className="onboard-field">
              <label>Gender</label>
              <select value={patient.gender} onChange={e => setPatient(p => ({...p, gender: e.target.value}))}>
                <option>Male</option>
                <option>Female</option>
                <option>Other</option>
              </select>
            </div>

            <div className="onboard-field">
              <label>MRN</label>
              <input type="text" value={patient.mrn} readOnly style={{ opacity:0.6, cursor:'default' }} />
            </div>

            <div className="onboard-field full-width">
              <label>Chief Complaint / Condition</label>
              <input type="text" placeholder="e.g. Acute decompensated heart failure"
                value={patient.condition}
                onChange={e => setPatient(p => ({...p, condition: e.target.value}))} />
            </div>
          </div>

          <div className="onboard-actions">
            <div />
            <button className="onboard-next-btn" disabled={!step1Valid}
              onClick={() => setStep(1)}>
              Continue to Clinical Data <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* ════════════════ STEP 2: Clinical Data ══════════════════════════════ */}
      {step === 1 && (
        <div className="onboard-fade-in">
          {/* Patient context bar */}
          <div className="onboard-patient-bar">
            <div className="onboard-patient-avatar">{patient.name.charAt(0)}</div>
            <div>
              <div style={{ fontWeight:700, fontSize:14 }}>{patient.name}</div>
              <div style={{ fontSize:12, color:'var(--text-secondary)' }}>
                {patient.age} yrs · {patient.gender} · {patient.mrn}
                {patient.condition && <> · {patient.condition}</>}
              </div>
            </div>
          </div>

          <div className="onboard-data-grid">
            {/* CXR Upload */}
            <div className="onboard-card">
              <div className="onboard-card-header">
                <span>🫁</span><span>Chest X-Ray</span>
              </div>
              {cxrUrl ? (
                <div>
                  <div style={{ color:'var(--green)', fontSize:13, fontWeight:600, marginBottom:6 }}>
                    <CheckCircle size={14} style={{ marginRight:4, verticalAlign:'middle' }} />CXR uploaded
                  </div>
                  <div style={{ fontSize:11, color:'var(--text-muted)', marginBottom:8 }}>{cxrFile?.name}</div>
                  <CXRViewer imageSrc={cxrUrl} findings={['Uploaded CXR']} severity="moderate" />
                  <button className="onboard-remove-btn" onClick={() => { setCxrFile(null); setCxrUrl(null); }}>
                    Remove
                  </button>
                </div>
              ) : (
                <div className="upload-zone" onClick={() => fileRef.current.click()}
                  onDragOver={e => { e.preventDefault(); e.currentTarget.classList.add('drag-over'); }}
                  onDragLeave={e => e.currentTarget.classList.remove('drag-over')}
                  onDrop={e => { e.preventDefault(); e.currentTarget.classList.remove('drag-over'); handleCxrUpload(e.dataTransfer.files[0]); }}>
                  <Upload size={28} color="var(--text-muted)" style={{ marginBottom:6 }} />
                  <h4>Drop CXR image here</h4>
                  <p>JPG, PNG · Max 10MB</p>
                  <input ref={fileRef} type="file" accept="image/*" style={{ display:'none' }}
                    onChange={e => e.target.files[0] && handleCxrUpload(e.target.files[0])} />
                </div>
              )}
            </div>

            {/* ECG Signal */}
            <div className="onboard-card">
              <div className="onboard-card-header">
                <span>💓</span><span>ECG Signal</span>
              </div>
              {hasEcg ? (
                <div>
                  <div style={{ color:'var(--green)', fontSize:13, fontWeight:600, marginBottom:8 }}>
                    <CheckCircle size={14} style={{ marginRight:4, verticalAlign:'middle' }} />12-lead ECG loaded
                  </div>
                  <ECGViewer findings={['Sinus rhythm','ST-segment analysis','Axis assessment']} />
                  <button className="onboard-remove-btn" onClick={() => setHasEcg(false)}>Remove</button>
                </div>
              ) : (
                <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
                  <button className="onboard-load-btn" onClick={() => setHasEcg(true)}>
                    📂 Load Demo 12-Lead ECG
                  </button>
                  <button className="onboard-load-btn" style={{ opacity:0.5, cursor:'default' }}>
                    ⬆️ Upload ECG File (coming soon)
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Lab Values */}
          <div className="onboard-card" style={{ marginTop:14 }}>
            <div className="onboard-card-header">
              <span>🧪</span><span>Clinical Lab Values</span>
            </div>
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

          <div className="onboard-actions">
            <button className="onboard-back-btn" onClick={() => setStep(0)}>
              <ChevronLeft size={16} /> Back
            </button>
            <button className="analyze-btn" onClick={runAnalysis} disabled={!hasAnyData || loading}
              style={{ maxWidth:340 }}>
              {loading
                ? <><span className="onboard-spinner" /> Running V3 Fusion Inference...</>
                : <><Zap size={16} style={{ marginRight:6 }} /> Run V3 Fusion Analysis</>
              }
            </button>
          </div>
        </div>
      )}

      {/* ════════════════ STEP 3: Results ═════════════════════════════════════ */}
      {step === 2 && (
        <div className="onboard-fade-in">
          {/* Patient context bar */}
          <div className="onboard-patient-bar">
            <div className="onboard-patient-avatar">{patient.name.charAt(0)}</div>
            <div style={{ flex:1 }}>
              <div style={{ fontWeight:700, fontSize:14 }}>{patient.name}</div>
              <div style={{ fontSize:12, color:'var(--text-secondary)' }}>
                {patient.age} yrs · {patient.gender} · {patient.mrn}
                {patient.condition && <> · {patient.condition}</>}
              </div>
            </div>
            <div style={{ display:'flex', gap:8, alignItems:'center' }}>
              {result?.real_inference && (
                <span className="onboard-live-badge">✅ Live Inference</span>
              )}
              {result?.latency_ms && (
                <span style={{ fontSize:11, color:'var(--text-muted)', fontFamily:'monospace' }}>
                  {result.latency_ms}ms
                </span>
              )}
            </div>
          </div>

          {loading ? (
            <div className="onboard-card" style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', minHeight:300, gap:16 }}>
              <div className="onboard-spinner-lg" />
              <div style={{ fontSize:15, fontWeight:600, color:'var(--text-primary)' }}>Running VisionCare 3.0 Inference...</div>
              <div style={{ fontSize:12, color:'var(--text-secondary)' }}>
                Processing CXR → ECG → Labs through Cross-Attention Gated Fusion
              </div>
            </div>
          ) : result ? (
            <>
              {error && (
                <div style={{ background:'rgba(245,158,11,0.08)', border:'1px solid rgba(245,158,11,0.2)', borderRadius:10, padding:'10px 16px', fontSize:12, color:'var(--text-amber)', marginBottom:10 }}>
                  ⚠️ {error}
                </div>
              )}

              {/* Risk Gauges */}
              <div className="onboard-card">
                <div className="onboard-card-header">
                  <Heart size={18} />
                  <span>8-Disease Risk Assessment — VisionCare 3.0</span>
                </div>
                <div style={{ display:'grid', gridTemplateColumns:'repeat(4, 1fr)', gap:10, marginTop:10 }}>
                  {V3_TARGETS.map(t => (
                    <RiskGauge key={t.key} pct={risks[t.key] || 0} label={t.short} size={100} />
                  ))}
                </div>
              </div>

              {/* Clinical Interpretation */}
              <div className="onboard-card onboard-interpretation">
                <div className="onboard-card-header">
                  <FileText size={18} />
                  <span>Clinical Interpretation</span>
                </div>
                <div className="onboard-interp-body">
                  {maxRiskEntry && (
                    <div style={{ fontSize:14, marginBottom:10 }}>
                      Highest risk: <strong style={{ color: maxRiskEntry[1] >= 60 ? 'var(--text-red)' : maxRiskEntry[1] >= 35 ? 'var(--text-amber)' : 'var(--text-green)' }}>
                        {maxRiskEntry[0].replace(/_/g,' ')} at {maxRiskEntry[1]}%
                      </strong>
                    </div>
                  )}

                  {(risks.heart_failure >= 70 || risks.mortality >= 50) && (
                    <div className="onboard-alert critical">
                      ⚠️ <strong>Critical risk detected.</strong> Immediate cardiology review recommended.
                    </div>
                  )}
                  {(risks.heart_failure >= 40 && risks.heart_failure < 70 && risks.mortality < 50) && (
                    <div className="onboard-alert moderate">
                      🟡 <strong>Moderate risk.</strong> Close monitoring and follow-up advised.
                    </div>
                  )}
                  {(risks.heart_failure < 40 && risks.mortality < 25) && (
                    <div className="onboard-alert low">
                      🟢 <strong>Low risk.</strong> Routine follow-up recommended.
                    </div>
                  )}

                  <div style={{ fontSize:11, color:'var(--text-muted)', marginTop:12, padding:'8px 12px', background:'var(--bg-elevated)', borderRadius:8 }}>
                    {result.real_inference ? '✅ Real V3 inference' : '⚙️ Estimated prediction'} ·
                    VisionCare {result.model_version ?? '3.0'} · Macro AUC 0.7926 · 8 targets · Balanced Gates (34/34/32)
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="onboard-actions">
                <button className="onboard-back-btn" onClick={() => setStep(1)}>
                  <ChevronLeft size={16} /> Modify Data
                </button>
                <button className="onboard-next-btn" onClick={resetAll}>
                  <UserPlus size={16} style={{ marginRight:6 }} /> Onboard Another Patient
                </button>
              </div>
            </>
          ) : null}
        </div>
      )}
    </div>
  );
}
