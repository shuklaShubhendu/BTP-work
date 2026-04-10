// src/pages/EncounterAnalysis.jsx — VisionCare 3.0 (8 diseases, real inference)
import { useParams, useNavigate } from 'react-router-dom';
import { useState, useMemo } from 'react';
import { MOCK_PATIENTS, MOCK_ENCOUNTERS, V3_TARGETS } from '../utils/mockData';
import CXRViewer from '../components/CXRViewer';
import ECGViewer  from '../components/ECGViewer';
import RiskGauge  from '../components/RiskGauge';
import { Zap, RefreshCw } from 'lucide-react';
import { predictEncounter } from '../utils/api';

const labValClass = (status) => {
  const s = status?.toLowerCase();
  return s === 'critical' ? 'lab-val-critical' : s === 'high' ? 'lab-val-high' : s === 'low' ? 'lab-val-low' : s === 'borderline' ? 'lab-val-borderline' : 'lab-val-normal';
};
const labBadgeClass = (status) => {
  const s = status?.toLowerCase();
  return `lab-status-badge lsb-${s}`;
};

const CONTRIB_COLORS = { CXR:'#38bdf8', ECG:'#22c55e', Labs:'#a78bfa' };

export default function EncounterAnalysis() {
  const { patientId, encounterId } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab]     = useState('cxr');
  const [liveResult, setLiveResult]   = useState(null);   // real inference result
  const [inferring, setInferring]     = useState(false);
  const [inferError, setInferError]   = useState(null);

  const patient  = useMemo(() => MOCK_PATIENTS.find(p => p.id === patientId), [patientId]);
  const encounter = useMemo(() => MOCK_ENCOUNTERS[patientId]?.find(e => e.id === encounterId), [patientId, encounterId]);

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



  if (!patient || !encounter) return <div className="loading">Encounter not found.</div>;

  const gates = encounter.gates || { vision: 0.339, signal: 0.338, clinical: 0.323 };
  // Use live inference result if available, else fall back to cached encounter risks
  const risks = liveResult?.risks || encounter.risks || {};
  const isLive = !!liveResult?.real_inference;

  // Find highest risk for alert
  const maxRisk = Math.max(...Object.values(risks));

  return (
    <div className="encounter-grid" style={{ height: 'calc(100vh - 60px)' }}>

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
          <CXRViewer imageSrc={encounter.cxr_image} findings={encounter.cxr_findings} severity={patient.severity} />
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

          {/* 8 Disease Gauges in 2x4 grid */}
          <div style={{
            display:'grid',
            gridTemplateColumns:'repeat(4, 1fr)',
            gap: 8,
            padding: '4px 0'
          }}>
            {V3_TARGETS.map(t => (
              <RiskGauge
                key={t.key}
                pct={risks[t.key] || 0}
                label={t.short}
                size={100}
              />
            ))}
          </div>

          <div className="model-badge-row">
            <Zap size={13} />
            VisionCare 3.0 · ResNet-50 + 1D ResNet-18 + 3-NN
            {isLive
              ? <span style={{ marginLeft:8, color:'var(--green)', fontWeight:700 }}>✅ Live Inference</span>
              : <span style={{ marginLeft:8, color:'var(--text-muted)' }}>· Cached Results</span>}
          </div>
          <div className="pred-time-row">
            {liveResult
              ? <>Live run at {new Date().toLocaleTimeString()} · {liveResult.latency_ms}ms · {liveResult.inputs_used?.labs || 0} labs parsed</>
              : <>Predicted at {encounter.date} · 15:00 UTC</>}
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
