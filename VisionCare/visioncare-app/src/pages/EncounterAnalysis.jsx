// src/pages/EncounterAnalysis.jsx — VisionCare 3.0 (8 diseases, real inference)
import { useParams, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import CXRViewer from '../components/CXRViewer';
import ECGViewer  from '../components/ECGViewer';
import { Zap, RefreshCw } from 'lucide-react';
import { getEncounter, getPatient, predictEncounter } from '../utils/api';
import { PageError, PageLoader } from '../components/PageState';

const labValClass = (status) => {
  const s = status?.toLowerCase();
  return s === 'critical' ? 'lab-val-critical' : s === 'high' ? 'lab-val-high' : s === 'low' ? 'lab-val-low' : s === 'borderline' ? 'lab-val-borderline' : 'lab-val-normal';
};
const labBadgeClass = (status) => {
  const s = status?.toLowerCase();
  return `lab-status-badge lsb-${s}`;
};

const TARGET_GROUPS = [
  { key: 'heart_failure', label: 'Heart Failure', why: 'Driven most by BNP elevation, congestion on CXR, and chronic structural stress.' },
  { key: 'mortality', label: 'Mortality', why: 'Captures overall acuity using combined organ stress and instability signals.' },
  { key: 'myocardial_infarction', label: 'Myocardial Infarction', why: 'Reflects acute ischemic injury pattern, especially troponin and ECG changes.' },
  { key: 'arrhythmia', label: 'Arrhythmia', why: 'Linked to rhythm instability, conduction changes, and underlying chamber strain.' },
  { key: 'sepsis', label: 'Sepsis', why: 'Represents systemic inflammatory decompensation risk from labs and physiologic stress.' },
  { key: 'pulmonary_embolism', label: 'Pulmonary Embolism', why: 'Highlights thromboembolic concern when dyspnea and cardiopulmonary strain align.' },
  { key: 'acute_kidney_injury', label: 'AKI', why: 'Usually tracks cardiorenal stress, renal hypoperfusion, and metabolic burden.' },
  { key: 'icu_admission', label: 'ICU Admission', why: 'Summarizes how likely the overall presentation is to need critical care escalation.' },
];

function getRiskTone(value) {
  if (value >= 70) return { label: 'Critical', color: 'var(--text-red)', bg: 'rgba(239,68,68,0.10)', border: 'rgba(239,68,68,0.22)' };
  if (value >= 40) return { label: 'Moderate', color: 'var(--text-amber)', bg: 'rgba(245,158,11,0.10)', border: 'rgba(245,158,11,0.22)' };
  return { label: 'Low', color: 'var(--text-green)', bg: 'rgba(34,197,94,0.10)', border: 'rgba(34,197,94,0.22)' };
}

export default function EncounterAnalysis() {
  const { patientId, encounterId } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab]     = useState('cxr');
  const [liveResult, setLiveResult]   = useState(null);   // real inference result
  const [inferring, setInferring]     = useState(false);
  const [inferError, setInferError]   = useState(null);
  const [patient, setPatient]         = useState(null);
  const [encounter, setEncounter]     = useState(null);
  const [loading, setLoading]         = useState(true);
  const [loadError, setLoadError]     = useState('');

  useEffect(() => {
    let active = true;
    async function load() {
      setLoading(true);
      setLoadError('');
      try {
        const [patientData, encounterData] = await Promise.all([
          getPatient(patientId),
          getEncounter(patientId, encounterId),
        ]);
        if (!active) return;
        setPatient(patientData);
        setEncounter(encounterData);
      } catch (err) {
        if (!active) return;
        setLoadError('We could not load this encounter from the backend.');
      } finally {
        if (active) setLoading(false);
      }
    }
    load();
    return () => { active = false; };
  }, [patientId, encounterId]);

  const runLiveInference = async () => {
    setInferring(true);
    setInferError(null);
    try {
      const res = await predictEncounter(encounterId);
      setLiveResult(res);
    } catch (err) {
      setInferError('Backend unavailable — showing cached results.');
    } finally {
      setInferring(false);
    }
  };



  if (loading) {
    return <div className="page-content"><PageLoader title="Loading encounter..." subtitle="Retrieving imaging, labs, and saved predictions for this patient." /></div>;
  }

  if (loadError || !patient || !encounter) {
    return (
      <div className="page-content">
        <PageError
          title="Encounter unavailable"
          subtitle={loadError || 'This encounter could not be found.'}
          action={<button className="state-action-btn" onClick={() => navigate(`/patients/${patientId}`)}>Back to Patient</button>}
        />
      </div>
    );
  }

  // Use live inference result if available, else fall back to cached encounter risks
  const risks = liveResult?.risks || encounter.risks || {};
  const isLive = !!liveResult?.real_inference;
  const analysisSource = liveResult?.analysis_source || encounter.analysis_source || (isLive ? 'model' : 'seeded');

  // Find highest risk for alert
  const maxRisk = Math.max(...Object.values(risks));
  const rankedTargets = TARGET_GROUPS
    .map(item => ({ ...item, value: risks[item.key] || 0, tone: getRiskTone(risks[item.key] || 0) }))
    .sort((a, b) => b.value - a.value);
  const topThree = rankedTargets.slice(0, 3);

  return (
    <div className="encounter-grid" style={{ height: 'calc(100vh - 60px)', gridTemplateColumns: '1fr 380px' }}>

      {/* ── LEFT PANEL: Viewer ── */}
      <div className="enc-panel" style={{ background:'#07111f' }}>
        {/* Breadcrumb */}
        <div style={{ padding:'10px 14px', borderBottom:'1px solid var(--border)', display:'flex', alignItems:'center', gap:6, fontSize:12, color:'var(--text-secondary)' }}>
          <span className="bc-link" onClick={() => navigate('/dashboard')}>Dashboard</span>
          <span>›</span>
          <span className="bc-link" onClick={() => navigate(`/patients/${patientId}`)}>{patient.name}</span>
          <span>›</span>
          <span style={{ color:'var(--text-primary)', fontWeight:600 }}>{encounter.label}</span>
        </div>

        {/* Tabs */}
        <div className="mod-tabs">
          {[['cxr','CXR'],['ecg','ECG'],['labs','Labs']].map(([key, label]) => (
            <button key={key} className={`mod-tab${activeTab === key ? ' active' : ''}`} onClick={() => setActiveTab(key)}>
              <span className="mod-dot" />
              {label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {activeTab === 'cxr' && (
          <CXRViewer imageSrc={encounter.cxr_image_url || encounter.cxr_image} findings={encounter.cxr_findings} severity={patient.severity} />
        )}
        {activeTab === 'ecg' && (
          <ECGViewer findings={encounter.ecg_findings} />
        )}
        {activeTab === 'labs' && (
          <div className="labs-area">
            <table className="labs-tbl">
              <thead><tr><th>Lab Name</th><th>Value</th><th>Normal</th><th>Status</th></tr></thead>
              <tbody>
                {encounter.labs?.map((lab, i) => (
                  <tr key={i}>
                    <td>{lab.name}</td>
                    <td className={labValClass(lab.status)}>{lab.value}</td>
                    <td style={{ color:'var(--text-muted)', fontSize:12 }}>{lab.normal}</td>
                    <td><span className={labBadgeClass(lab.status)}>{lab.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── MIDDLE PANEL: V3 Risk Assessment (8 diseases) ── */}
      <div className="enc-panel" style={{ background:'var(--bg-card)', overflowY:'auto' }}>
        <div className="risk-section">
          <div className="risk-label">8-DISEASE RISK ASSESSMENT</div>

          <div style={{
            display:'grid',
            gridTemplateColumns:'repeat(2, minmax(0, 1fr))',
            gap: 10,
            padding: '4px 0'
          }}>
            {rankedTargets.map(item => (
              <div key={item.key} style={{ border:`1px solid ${item.tone.border}`, background:item.tone.bg, borderRadius:12, padding:'12px 14px' }}>
                <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', gap:10 }}>
                  <div style={{ fontSize:13, fontWeight:700 }}>{item.label}</div>
                  <div style={{ color:item.tone.color, fontSize:20, fontWeight:800 }}>{item.value}%</div>
                </div>
                <div style={{ marginTop:4, fontSize:11, color:item.tone.color, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.06em' }}>
                  {item.tone.label} Priority
                </div>
                <div style={{ marginTop:8, fontSize:11, color:'var(--text-secondary)', lineHeight:1.5 }}>
                  {item.why}
                </div>
              </div>
            ))}
          </div>

          <div className="model-badge-row">
            <Zap size={13} />
            VisionCare 3.0 · ResNet-50 + 1D ResNet-18 + 3-NN
            {analysisSource === 'model'
              ? <span style={{ marginLeft:8, color:'var(--green)', fontWeight:700 }}>✅ Live Inference</span>
              : analysisSource === 'deterministic_estimate'
                ? <span style={{ marginLeft:8, color:'var(--text-amber)', fontWeight:700 }}>· Stable Estimated Result</span>
                : <span style={{ marginLeft:8, color:'var(--text-muted)' }}>· Stored Demo Record</span>}
          </div>
          <div className="pred-time-row">
            {liveResult
              ? <>Refreshed at {liveResult.last_inference_at || new Date().toLocaleTimeString()} · {liveResult.latency_ms}ms · {liveResult.inputs_used?.labs || 0} labs parsed</>
              : <>Recorded encounter date {encounter.date}{encounter.last_inference_at ? ` · last refreshed ${encounter.last_inference_at}` : ''}</>}
          </div>

          {/* Live Inference Button */}
          <div style={{ marginTop:10 }}>
            <button
              onClick={runLiveInference}
              disabled={inferring}
              style={{ width:'100%', padding:'9px', background: inferring ? 'var(--bg-elevated)' : 'var(--green-dim)',
                border:'1px solid var(--green-border)', borderRadius:8,
                color: inferring ? 'var(--text-muted)' : 'var(--green)',
                fontSize:12, fontWeight:700, cursor: inferring ? 'default' : 'pointer',
                display:'flex', alignItems:'center', justifyContent:'center', gap:6, transition:'all 0.15s' }}>
              {inferring
                ? <><RefreshCw size={13} style={{ animation:'spin 1s linear infinite' }} /> Running V3 Inference...</>
                : <><Zap size={13} /> Run Live V3 Inference</>}
            </button>
            {inferError && <div style={{ fontSize:11, color:'var(--text-amber)', marginTop:4, textAlign:'center' }}>{inferError}</div>}
          </div>

          <div className="patient-info-box">
            {[['Name', patient.name], ['MRN', patient.mrn], ['Age / Gender', `${patient.age} / ${patient.gender}`], ['Encounter', encounter.label]].map(([k, v]) => (
              <div key={k} className="info-row">
                <span className="info-key">{k}</span>
                <span className="info-val">{v}</span>
              </div>
            ))}
          </div>

          <div style={{ background:'var(--bg-elevated)', borderRadius:10, padding:'12px 14px' }}>
            <div style={{ fontSize:10, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:10 }}>Clinical Summary</div>
            <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
              {topThree.map(item => (
                <div key={item.key} style={{ border:`1px solid ${item.tone.border}`, background:item.tone.bg, borderRadius:10, padding:'10px 12px' }}>
                  <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', gap:10 }}>
                    <div style={{ fontSize:13, fontWeight:700 }}>{item.label}</div>
                    <div style={{ color:item.tone.color, fontSize:12, fontWeight:800 }}>{item.value}% · {item.tone.label}</div>
                  </div>
                  <div style={{ marginTop:6, fontSize:11, color:'var(--text-secondary)', lineHeight:1.5 }}>{item.why}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ background:'var(--bg-elevated)', borderRadius:10, padding:'12px 14px' }}>
            <div style={{ fontSize:10, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:10 }}>Confidence Notes</div>
            <div style={{ fontSize:12, color:'var(--text-secondary)', lineHeight:1.7 }}>
              Highest concern is <strong style={{ color: topThree[0]?.tone.color }}>{topThree[0]?.label}</strong> at <strong style={{ color: topThree[0]?.tone.color }}>{topThree[0]?.value}%</strong>.
              {' '}{analysisSource === 'model'
                ? 'These values come from the loaded V3 model using the stored encounter inputs.'
                : analysisSource === 'deterministic_estimate'
                  ? 'These values are a stable deterministic estimate generated from the stored encounter inputs because the full model was not available.'
                  : 'These values are part of the seeded encounter record and have not been refreshed yet.'}
            </div>
          </div>

          {/* Alert banner for high risk */}
          {maxRisk >= 70 && (
            <div style={{ background:'rgba(239,68,68,0.1)', border:'1px solid rgba(239,68,68,0.25)', borderRadius:8, padding:'10px 12px', fontSize:12, color:'var(--text-red)' }}>
              ⚠️ <strong>Critical Alert</strong> — One or more disease risks exceed 70%. Immediate clinical review recommended.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
