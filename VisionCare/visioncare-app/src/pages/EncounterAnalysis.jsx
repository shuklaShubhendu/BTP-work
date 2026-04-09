// src/pages/EncounterAnalysis.jsx
import { useParams, useNavigate } from 'react-router-dom';
import { useState, useMemo } from 'react';
import { MOCK_PATIENTS, MOCK_ENCOUNTERS } from '../utils/mockData';
import CXRViewer from '../components/CXRViewer';
import ECGViewer  from '../components/ECGViewer';
import RiskGauge  from '../components/RiskGauge';
import AIChat     from '../components/AIChat';
import { Zap } from 'lucide-react';

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
  const [activeTab, setActiveTab] = useState('cxr');

  const patient  = useMemo(() => MOCK_PATIENTS.find(p => p.id === patientId), [patientId]);
  const encounter = useMemo(() => MOCK_ENCOUNTERS[patientId]?.find(e => e.id === encounterId), [patientId, encounterId]);

  if (!patient || !encounter) return <div className="loading">Encounter not found.</div>;

  const gates = encounter.gates || { vision: 0.33, signal: 0.33, clinical: 0.34 };
  const contribs = [
    { name:'CXR',  pct: Math.round(gates.vision   * 100), color: CONTRIB_COLORS.CXR  },
    { name:'ECG',  pct: Math.round(gates.signal   * 100), color: CONTRIB_COLORS.ECG  },
    { name:'Labs', pct: Math.round(gates.clinical * 100), color: CONTRIB_COLORS.Labs },
  ];

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

      {/* ── MIDDLE PANEL: Risk Assessment ── */}
      <div className="enc-panel" style={{ background:'var(--bg-card)' }}>
        <div className="risk-section">
          <div className="risk-label">RISK ASSESSMENT</div>
          <div className="gauges-row" style={{ justifyContent:'center' }}>
            <RiskGauge pct={encounter.hf_risk}          label="Heart Failure Risk" />
            <RiskGauge pct={encounter.mortality_risk}   label="Mortality Risk" />
          </div>

          <div className="divider" />

          <div className="contrib-section">
            <div className="contrib-title">MODALITY CONTRIBUTION</div>
            {contribs.map(c => (
              <div key={c.name} className="contrib-row">
                <span className="contrib-name" style={{ color:c.color }}>{c.name}</span>
                <div className="contrib-track">
                  <div className="contrib-fill" style={{ width:`${c.pct}%`, background:c.color }} />
                </div>
                <span className="contrib-pct" style={{ color:c.color }}>{c.pct}%</span>
              </div>
            ))}
          </div>

          <div className="model-badge-row">
            <Zap size={13} />
            VisionCare 2.0 Fusion · 1.4s latency
          </div>
          <div className="pred-time-row">Predicted at {encounter.date} · 15:00 UTC</div>

          <div className="patient-info-box">
            {[['Name', patient.name], ['MRN', patient.mrn], ['Age / Gender', `${patient.age} / ${patient.gender}`], ['Encounter', encounter.label]].map(([k, v]) => (
              <div key={k} className="info-row">
                <span className="info-key">{k}</span>
                <span className="info-val">{v}</span>
              </div>
            ))}
          </div>

          {/* Alert banner for critical */}
          {patient.severity === 'critical' && (
            <div style={{ background:'rgba(239,68,68,0.1)', border:'1px solid rgba(239,68,68,0.25)', borderRadius:8, padding:'10px 12px', fontSize:12, color:'var(--text-red)' }}>
              ⚠️ <strong>Critical Alert</strong> — Immediate clinical review recommended. HF risk &gt;70%.
            </div>
          )}
        </div>
      </div>

      {/* ── RIGHT PANEL: AI Chat ── */}
      <div className="enc-panel" style={{ overflow:'hidden' }}>
        <AIChat encounter={encounter} patient={patient} />
      </div>
    </div>
  );
}
