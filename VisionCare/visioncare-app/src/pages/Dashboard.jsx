// src/pages/Dashboard.jsx
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, Heart, AlertTriangle, Activity } from 'lucide-react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { MOCK_PATIENTS } from '../utils/mockData';

const SEVERITY_DATA = [
  { name: 'Critical', value: 3, color: '#ef4444' },
  { name: 'Moderate', value: 3, color: '#f59e0b' },
  { name: 'Normal',   value: 2, color: '#22c55e' },
];

const TREND_DATA = [
  { day: 'Mon', predictions: 72, critical: 18 },
  { day: 'Tue', predictions: 89, critical: 24 },
  { day: 'Wed', predictions: 65, critical: 15 },
  { day: 'Thu', predictions: 93, critical: 27 },
  { day: 'Fri', predictions: 81, critical: 21 },
  { day: 'Sat', predictions: 55, critical: 12 },
  { day: 'Sun', predictions: 89, critical: 26 },
];

const PERF_DATA = [
  { metric: 'Mortality', v2: 0.80, v3: 0.83 },
  { metric: 'HF', v2: 0.82, v3: 0.75 },
  { metric: 'Macro', v2: 0.81, v3: 0.79 },
];

function SeverityDot({ s }) {
  const cls = s === 'critical' ? 'dot-critical' : s === 'moderate' ? 'dot-moderate' : 'dot-normal';
  return <span className={`severity-dot ${cls}`} />;
}

function getBadgeClass(hf) {
  return hf >= 70 ? 'badge badge-critical' : hf >= 40 ? 'badge badge-high' : 'badge badge-low';
}
function getBadgeLabel(hf) {
  return hf >= 70 ? 'High' : hf >= 40 ? 'Moderate' : 'Low';
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [patients, setPatients] = useState(MOCK_PATIENTS);

  const STATS = [
    { label:'TOTAL PATIENTS',    val:'10,842', delta:'+124 this month', icon:<Users size={16}/>,         cls:'si-green'  },
    { label:'HIGH RISK HF',      val:'2,341',  delta:'+12 today',       icon:<Heart size={16}/>,         cls:'si-red'    },
    { label:'CRITICAL CASES',    val:'156',    delta:'-3 from yesterday',icon:<AlertTriangle size={16}/>, cls:'si-amber'  },
    { label:'PREDICTIONS TODAY', val:'89',     delta:'Avg 1.4s latency', icon:<Activity size={16}/>,      cls:'si-blue'   },
  ];

  const tooltipStyle = { background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#f1f5f9', fontSize: 12 };

  return (
    <div className="page-content section-gap">
      {/* Stat Cards */}
      <div className="stat-grid">
        {STATS.map(s => (
          <div key={s.label} className="stat-card">
            <div className="stat-top">
              <span className="stat-label">{s.label}</span>
              <div className={`stat-icon ${s.cls}`}>{s.icon}</div>
            </div>
            <div className="stat-val">{s.val}</div>
            <div className="stat-delta">{s.delta}</div>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div className="chart-section">
        {/* Predictions trend */}
        <div className="card">
          <div className="card-title">Weekly Predictions</div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={TREND_DATA}>
              <XAxis dataKey="day" tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="predictions" stroke="#22c55e" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="critical"    stroke="#ef4444" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ display:'flex', gap:16, marginTop:8 }}>
            <span style={{ fontSize:11, color:'#22c55e' }}>● All Predictions</span>
            <span style={{ fontSize:11, color:'#ef4444' }}>● Critical Cases</span>
          </div>
        </div>

        {/* Severity distribution */}
        <div className="card" style={{ display:'flex', gap:16, alignItems:'center' }}>
          <div>
            <div className="card-title">Patient Severity</div>
            <PieChart width={120} height={120}>
              <Pie data={SEVERITY_DATA} cx={55} cy={55} innerRadius={36} outerRadius={55} dataKey="value" strokeWidth={0}>
                {SEVERITY_DATA.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Pie>
            </PieChart>
          </div>
          <div style={{ flex:1, display:'flex', flexDirection:'column', gap:10 }}>
            {SEVERITY_DATA.map(d => (
              <div key={d.name} style={{ display:'flex', alignItems:'center', gap:8 }}>
                <div style={{ width:10,height:10,borderRadius:'50%',background:d.color,flexShrink:0 }}/>
                <span style={{ fontSize:12,flex:1,color:'var(--text-secondary)' }}>{d.name}</span>
                <span style={{ fontSize:13,fontWeight:700,color:d.color }}>{d.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model Performance mini-bar */}
      <div className="card">
        <div className="card-title">Model AUC Performance — V2 vs V3</div>
        <ResponsiveContainer width="100%" height={100}>
          <BarChart data={PERF_DATA} layout="vertical" barCategoryGap="20%">
            <XAxis type="number" domain={[0.7, 0.9]} tick={{ fill:'#64748b', fontSize:10 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(2)} />
            <YAxis type="category" dataKey="metric" tick={{ fill:'#94a3b8', fontSize:11 }} axisLine={false} tickLine={false} width={60} />
            <Tooltip contentStyle={tooltipStyle} formatter={v => v.toFixed(4)} />
            <Bar dataKey="v2" fill="#22c55e" radius={[0,4,4,0]} name="V2 (Active)" />
            <Bar dataKey="v3" fill="#3b82f6" radius={[0,4,4,0]} name="V3 (Training)" />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ display:'flex', gap:16, marginTop:8 }}>
          <span style={{ fontSize:11, color:'#22c55e' }}>■ V2 Active (AUC 0.8105)</span>
          <span style={{ fontSize:11, color:'#3b82f6' }}>■ V3 In Training</span>
        </div>
      </div>

      {/* Recent Patients Table */}
      <div className="card" style={{ overflowX:'auto' }}>
        <div className="card-title">Recent Patients</div>
        <table className="pt-table">
          <thead>
            <tr>
              <th>MRN</th><th>PATIENT NAME</th><th>AGE / GENDER</th>
              <th>SEVERITY</th><th>LAST ENCOUNTER</th><th>HF RISK</th><th>MORTALITY RISK</th><th>STATUS</th>
            </tr>
          </thead>
          <tbody>
            {patients.map(p => {
              const enc = { hf_risk: p.hf_risk, mortality_risk: p.mortality_risk };
              return (
                <tr key={p.id} onClick={() => navigate(`/patients/${p.id}`)}>
                  <td className="mrn-cell">{p.mrn}</td>
                  <td><strong>{p.name}</strong></td>
                  <td>{p.age} / {p.gender[0]}</td>
                  <td><SeverityDot s={p.severity} />{p.severity.charAt(0).toUpperCase() + p.severity.slice(1)}</td>
                  <td style={{ color:'var(--text-secondary)',fontSize:12 }}>{p.hf_risk >= 70 ? '26 Mar 2026' : p.hf_risk >= 40 ? '22 Mar 2026' : '15 Mar 2026'}</td>
                  <td><span className={getBadgeClass(p.hf_risk)}>{p.hf_risk}% {getBadgeLabel(p.hf_risk)}</span></td>
                  <td><span className={getBadgeClass(p.mortality_risk)}>{p.mortality_risk}% {getBadgeLabel(p.mortality_risk)}</span></td>
                  <td className={p.status === 'Active' ? 'status-active' : 'status-discharged'}>
                    {p.status === 'Active' ? '● Active' : '○ Discharged'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
