// src/pages/PatientProfile.jsx
import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Calendar } from 'lucide-react';
import { getEncounters, getPatient } from '../utils/api';
import { EmptyState, PageError, PageLoader } from '../components/PageState';

export default function PatientProfile() {
  const { patientId } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState(null);
  const [encounters, setEncounters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    async function load() {
      setLoading(true);
      setError('');
      try {
        const [patientData, encounterData] = await Promise.all([
          getPatient(patientId),
          getEncounters(patientId),
        ]);
        if (!active) return;
        setPatient(patientData);
        setEncounters(encounterData);
      } catch (err) {
        if (!active) return;
        setError('We could not load this patient from the backend.');
      } finally {
        if (active) setLoading(false);
      }
    }
    load();
    return () => { active = false; };
  }, [patientId]);

  if (loading) {
    return <div className="page-content"><PageLoader title="Loading patient profile..." subtitle="Syncing demographics, encounters, and latest risk information." /></div>;
  }

  if (error || !patient) {
    return (
      <div className="page-content">
        <PageError
          title="Patient not found"
          subtitle={error || 'This patient could not be loaded.'}
          action={<button className="state-action-btn" onClick={() => navigate('/patients')}>Back to Patients</button>}
        />
      </div>
    );
  }

  const getBadge = (hf) => {
    const cls = hf >= 70 ? 'badge-critical' : hf >= 40 ? 'badge-high' : 'badge-low';
    const lbl = hf >= 70 ? 'High' : hf >= 40 ? 'Moderate' : 'Low';
    return <span className={`badge ${cls}`}>{hf}% HF</span>;
  };
  const getMortBadge = (m) => {
    const cls = m >= 40 ? 'badge-critical' : m >= 20 ? 'badge-high' : 'badge-low';
    return <span className={`badge ${cls}`}>{m}% Mort.</span>;
  };

  return (
    <div className="page-content section-gap">
      <div className="breadcrumb">
        <span className="bc-link" onClick={() => navigate('/dashboard')}>Dashboard</span>
        <span className="bc-sep">›</span>
        <span className="bc-current">{patient.name}</span>
      </div>

      {/* Patient header */}
      <div className="pat-header-card">
        <div>
          <div className="pat-name-big">{patient.name}</div>
          <div className="pat-mrn">{patient.mrn}</div>
        </div>
        <div className="pat-meta">
          <div className="meta-item">
            <span className="meta-key">AGE</span>
            <span className="meta-val">{patient.age}</span>
          </div>
          <div className="meta-item">
            <span className="meta-key">GENDER</span>
            <span className="meta-val">{patient.gender}</span>
          </div>
          <div className="meta-item">
            <span className="meta-key">CONDITION</span>
            <span className="meta-val" style={{ fontSize:13, maxWidth:180, textAlign:'center' }}>{patient.condition}</span>
          </div>
          <div className="meta-item">
            <span className="meta-key">STATUS</span>
            <span className="meta-val" style={{ color: patient.status === 'Active' ? 'var(--green)' : 'var(--text-secondary)' }}>
              {patient.status}
            </span>
          </div>
        </div>
      </div>

      {/* Severity alert */}
      {patient.severity === 'critical' && (
        <div style={{ background:'rgba(239,68,68,0.08)', border:'1px solid rgba(239,68,68,0.2)', borderRadius:10, padding:'12px 16px', fontSize:13, color:'var(--text-red)', display:'flex', alignItems:'center', gap:10 }}>
          <span style={{ fontSize:18 }}>⚠️</span>
          <span><strong>High Priority Patient</strong> — Latest encounter shows elevated multi-disease risks. Immediate clinical review recommended per AHA 2022 guidelines.</span>
        </div>
      )}

      {/* Encounter History */}
      <div>
        <div className="card-title" style={{ marginBottom:12 }}>Encounter History</div>
        {encounters.length === 0 ? (
          <EmptyState title="No encounters yet" subtitle="This patient has been created, but no encounter analysis has been saved yet." />
        ) : (
        <div className="section-gap">
          {encounters.map(enc => (
            <div key={enc.id} className="enc-card" onClick={() => navigate(`/patients/${patientId}/encounters/${enc.id}`)}>
              <div className="enc-icon">
                <Calendar size={17} color="var(--text-secondary)" />
              </div>
              <div className="enc-info">
                <div className="enc-label-text">{enc.label} · {enc.description}</div>
                <div className="enc-date">Admitted {enc.date}</div>
              </div>
              <div className="enc-risks-row">
                {getBadge(enc.risks?.heart_failure || enc.hf_risk || 0)}
                {getMortBadge(enc.risks?.mortality || enc.mortality_risk || 0)}
              </div>
            </div>
          ))}
        </div>
        )}
      </div>

      {/* Quick stats from latest encounter */}
      {encounters[0] && (
        <div className="card">
          <div className="card-title">Latest Encounter Summary — {encounters[0].label}</div>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:14 }}>
            {[['Macro AUC', '0.7926', 'var(--green)'],
              ['Disease Targets', '8', '#38bdf8'],
              ['Best Epoch', '17/25', '#a78bfa']
            ].map(([label, val, color]) => (
              <div key={label} style={{ background:'var(--bg-elevated)', borderRadius:8, padding:'14px 16px', textAlign:'center' }}>
                <div style={{ fontSize:10, color:'var(--text-muted)', marginBottom:4, textTransform:'uppercase', letterSpacing:'0.05em' }}>{label}</div>
                <div style={{ fontSize:26, fontWeight:800, color }}>{val}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop:14, fontSize:12, color:'var(--text-secondary)' }}>
            <strong style={{ color:'var(--text-primary)' }}>CXR findings:</strong> {encounters[0].cxr_findings?.join(', ')}
          </div>
          <div style={{ marginTop:6, fontSize:12, color:'var(--text-secondary)' }}>
            <strong style={{ color:'var(--text-primary)' }}>ECG findings:</strong> {encounters[0].ecg_findings?.join(', ')}
          </div>
        </div>
      )}
    </div>
  );
}
