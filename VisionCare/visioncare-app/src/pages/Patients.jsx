// src/pages/Patients.jsx
import { useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Search, Filter } from 'lucide-react';
import { MOCK_PATIENTS } from '../utils/mockData';

const SEV_COLOR = { critical:'#ef4444', moderate:'#f59e0b', normal:'#22c55e' };

function getBadge(hf) {
  const cls = hf >= 70 ? 'badge-critical' : hf >= 40 ? 'badge-high' : 'badge-low';
  const lbl = hf >= 70 ? 'High' : hf >= 40 ? 'Moderate' : 'Low';
  return <span className={`badge ${cls}`}>{hf}% {lbl}</span>;
}

export default function Patients() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [q, setQ] = useState(searchParams.get('q') || '');
  const [sevFilter, setSevFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');

  const filtered = MOCK_PATIENTS.filter(p => {
    const matchQ = !q || p.name.toLowerCase().includes(q.toLowerCase()) || p.mrn.toLowerCase().includes(q.toLowerCase());
    const matchSev = sevFilter === 'all' || p.severity === sevFilter;
    const matchStatus = statusFilter === 'all' || p.status.toLowerCase() === statusFilter;
    return matchQ && matchSev && matchStatus;
  });

  return (
    <div className="page-content section-gap">
      {/* Filters */}
      <div className="card" style={{ padding:'14px 18px' }}>
        <div style={{ display:'flex', gap:12, alignItems:'center', flexWrap:'wrap' }}>
          <div className="search-box" style={{ flex:1, minWidth:200 }}>
            <Search size={14} color="var(--text-muted)" />
            <input placeholder="Search by name or MRN..." value={q} onChange={e => setQ(e.target.value)} />
          </div>
          <div style={{ display:'flex', gap:8, alignItems:'center' }}>
            <Filter size={14} color="var(--text-muted)" />
            <select value={sevFilter} onChange={e => setSevFilter(e.target.value)} style={{ background:'var(--bg-input)', border:'1px solid var(--border)', borderRadius:8, padding:'7px 10px', color:'var(--text-primary)', fontSize:13, outline:'none' }}>
              <option value="all">All Severity</option>
              <option value="critical">🔴 Critical</option>
              <option value="moderate">🟡 Moderate</option>
              <option value="normal">🟢 Normal</option>
            </select>
            <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)} style={{ background:'var(--bg-input)', border:'1px solid var(--border)', borderRadius:8, padding:'7px 10px', color:'var(--text-primary)', fontSize:13, outline:'none' }}>
              <option value="all">All Status</option>
              <option value="active">Active</option>
              <option value="discharged">Discharged</option>
            </select>
          </div>
          <span style={{ fontSize:12, color:'var(--text-secondary)' }}>{filtered.length} patients</span>
        </div>
      </div>

      {/* Severity summary pills */}
      <div style={{ display:'flex', gap:10 }}>
        {[['critical',3,'Critical — Immediate Review'],['moderate',3,'Moderate — Monitor'],['normal',2,'Normal — Routine']].map(([s,n,lbl]) => (
          <div key={s} onClick={() => setSevFilter(sevFilter === s ? 'all' : s)}
            style={{ flex:1, background:'var(--bg-card)', border:`1px solid ${sevFilter===s ? SEV_COLOR[s] : 'var(--border)'}`, borderRadius:10, padding:'12px 16px', cursor:'pointer', transition:'all 0.15s' }}>
            <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:4 }}>
              <div style={{ width:10,height:10,borderRadius:'50%',background:SEV_COLOR[s],boxShadow: s==='critical'?`0 0 8px ${SEV_COLOR[s]}`:'none' }} />
              <span style={{ fontSize:18, fontWeight:800, color:SEV_COLOR[s] }}>{n}</span>
            </div>
            <div style={{ fontSize:12, color:'var(--text-secondary)' }}>{lbl}</div>
          </div>
        ))}
      </div>

      {/* Patient cards */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(340px, 1fr))', gap:14 }}>
        {filtered.map(p => (
          <div key={p.id} className="card" onClick={() => navigate(`/patients/${p.id}`)}
            style={{ cursor:'pointer', transition:'all 0.15s', borderColor: p.severity === 'critical' ? 'rgba(239,68,68,0.2)' : 'var(--border)' }}
            onMouseEnter={e => e.currentTarget.style.borderColor = SEV_COLOR[p.severity]}
            onMouseLeave={e => e.currentTarget.style.borderColor = p.severity === 'critical' ? 'rgba(239,68,68,0.2)' : 'var(--border)'}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:12 }}>
              <div>
                <div style={{ fontSize:15, fontWeight:700 }}>{p.name}</div>
                <div style={{ fontSize:11, color:'var(--text-muted)', fontFamily:'monospace', marginTop:2 }}>{p.mrn}</div>
              </div>
              <div style={{ display:'flex', flexDirection:'column', alignItems:'flex-end', gap:4 }}>
                <span className={`badge badge-${p.severity === 'critical' ? 'critical' : p.severity === 'moderate' ? 'moderate' : 'normal'}`}>
                  {p.severity.charAt(0).toUpperCase() + p.severity.slice(1)}
                </span>
                <span style={{ fontSize:11, color: p.status === 'Active' ? 'var(--green)' : 'var(--text-muted)' }}>
                  {p.status === 'Active' ? '● Active' : '○ Discharged'}
                </span>
              </div>
            </div>

            <div style={{ fontSize:12, color:'var(--text-secondary)', marginBottom:12 }}>{p.age} yrs / {p.gender} · {p.condition}</div>

            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10 }}>
              <div style={{ background:'var(--bg-elevated)', borderRadius:8, padding:'8px 12px' }}>
                <div style={{ fontSize:10, color:'var(--text-muted)', marginBottom:2 }}>HF RISK</div>
                <div style={{ fontSize:18, fontWeight:800, color: p.hf_risk >= 70 ? 'var(--text-red)' : p.hf_risk >= 40 ? 'var(--text-amber)' : 'var(--text-green)' }}>{p.hf_risk}%</div>
              </div>
              <div style={{ background:'var(--bg-elevated)', borderRadius:8, padding:'8px 12px' }}>
                <div style={{ fontSize:10, color:'var(--text-muted)', marginBottom:2 }}>MORTALITY</div>
                <div style={{ fontSize:18, fontWeight:800, color: p.mortality_risk >= 40 ? 'var(--text-red)' : p.mortality_risk >= 20 ? 'var(--text-amber)' : 'var(--text-green)' }}>{p.mortality_risk}%</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
