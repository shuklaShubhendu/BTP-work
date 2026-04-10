// src/pages/Dashboard.jsx — VisionCare 3.0 Dashboard
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, Heart, AlertTriangle, Activity } from 'lucide-react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { getPatients, getStats } from '../utils/api';
import { PageError, PageLoader } from '../components/PageState';

const TREND_DATA = [
  { day: 'Mon', predictions: 72, critical: 18 },
  { day: 'Tue', predictions: 89, critical: 24 },
  { day: 'Wed', predictions: 65, critical: 15 },
  { day: 'Thu', predictions: 93, critical: 27 },
  { day: 'Fri', predictions: 81, critical: 21 },
  { day: 'Sat', predictions: 55, critical: 12 },
  { day: 'Sun', predictions: 89, critical: 26 },
];

// V3 per-disease AUC comparison
const V3_AUC_DATA = [
  { target: 'Sepsis',   auc: 0.8223 },
  { target: 'PE',       auc: 0.8213 },
  { target: 'Mortality',auc: 0.8082 },
  { target: 'Arrhyth.', auc: 0.7936 },
  { target: 'ICU',      auc: 0.7839 },
  { target: 'HF',       auc: 0.7829 },
  { target: 'AKI',      auc: 0.7813 },
  { target: 'MI',       auc: 0.7473 },
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
  const [patients, setPatients] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    async function load() {
      setLoading(true);
      setError('');
      try {
        const [statsData, patientsData] = await Promise.all([getStats(), getPatients()]);
        if (!active) return;
        setStats(statsData);
        setPatients(patientsData);
      } catch (err) {
        if (!active) return;
        setError('The dashboard could not reach the backend. Start the API and refresh to see live hospital data.');
      } finally {
        if (active) setLoading(false);
      }
    }
    load();
    return () => { active = false; };
  }, []);

  const STATS = [
    { label:'TOTAL PATIENTS',    val:String(stats?.total_patients ?? '--'), delta:'Live patient registry', icon:<Users size={16}/>,         cls:'si-green'  },
    { label:'HIGH RISK HF',      val:String(stats?.high_risk_hf ?? '--'),   delta:'Heart failure risk ≥ 70%', icon:<Heart size={16}/>,    cls:'si-red'    },
    { label:'DISEASE TARGETS',   val:String(stats?.num_targets ?? '8'),     delta:'V3: 8 diseases',  icon:<AlertTriangle size={16}/>, cls:'si-amber'  },
    { label:'PREDICTIONS TODAY', val:String(stats?.predictions_today ?? '--'), delta:'Encounter predictions available', icon:<Activity size={16}/>, cls:'si-blue'   },
  ];
  const severityData = [
    { name: 'Critical', value: patients.filter(p => p.severity === 'critical').length, color: '#ef4444' },
    { name: 'Moderate', value: patients.filter(p => p.severity === 'moderate').length, color: '#f59e0b' },
    { name: 'Normal', value: patients.filter(p => p.severity === 'normal').length, color: '#22c55e' },
  ];

  const tooltipStyle = { background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#f1f5f9', fontSize: 12 };

  if (loading) {
    return <div className="page-content"><PageLoader title="Loading dashboard..." subtitle="Pulling patient metrics, risk summaries, and recent encounters from the backend." /></div>;
  }

  if (error) {
    return (
      <div className="page-content">
        <PageError
          title="Dashboard unavailable"
          subtitle={error}
          action={<button className="state-action-btn" onClick={() => window.location.reload()}>Retry</button>}
        />
      </div>
    );
  }

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
              <Pie data={severityData} cx={55} cy={55} innerRadius={36} outerRadius={55} dataKey="value" strokeWidth={0}>
                {severityData.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Pie>
            </PieChart>
          </div>
          <div style={{ flex:1, display:'flex', flexDirection:'column', gap:10 }}>
            {severityData.map(d => (
              <div key={d.name} style={{ display:'flex', alignItems:'center', gap:8 }}>
                <div style={{ width:10,height:10,borderRadius:'50%',background:d.color,flexShrink:0 }}/>
                <span style={{ fontSize:12,flex:1,color:'var(--text-secondary)' }}>{d.name}</span>
                <span style={{ fontSize:13,fontWeight:700,color:d.color }}>{d.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* V3 Per-Disease AUC Bar Chart */}
      <div className="card">
        <div className="card-title">VisionCare 3.0 — Per-Disease AUC (8 Targets)</div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={V3_AUC_DATA} barCategoryGap="18%">
            <XAxis dataKey="target" tick={{ fill:'#94a3b8', fontSize:11 }} axisLine={false} tickLine={false} />
            <YAxis domain={[0.7, 0.85]} tick={{ fill:'#64748b', fontSize:10 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(2)} />
            <Tooltip contentStyle={tooltipStyle} formatter={v => v.toFixed(4)} />
            <Bar dataKey="auc" radius={[4,4,0,0]} name="AUC-ROC">
              {V3_AUC_DATA.map((entry, i) => (
                <Cell key={i} fill={entry.auc >= 0.80 ? '#22c55e' : entry.auc >= 0.78 ? '#3b82f6' : '#f59e0b'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div style={{ display:'flex', gap:16, marginTop:8, justifyContent:'center' }}>
          <span style={{ fontSize:11, color:'#22c55e' }}>■ AUC ≥ 0.80</span>
          <span style={{ fontSize:11, color:'#3b82f6' }}>■ AUC ≥ 0.78</span>
          <span style={{ fontSize:11, color:'#f59e0b' }}>■ AUC &lt; 0.78</span>
          <span style={{ fontSize:11, color:'var(--text-muted)' }}>| Macro AUC: 0.7926</span>
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
            {patients.map(p => (
              <tr key={p.id} onClick={() => navigate(`/patients/${p.id}`)}>
                <td className="mrn-cell">{p.mrn}</td>
                <td><strong>{p.name}</strong></td>
                <td>{p.age} / {p.gender[0]}</td>
                <td><SeverityDot s={p.severity} />{p.severity.charAt(0).toUpperCase() + p.severity.slice(1)}</td>
                <td style={{ color:'var(--text-secondary)',fontSize:12 }}>{p.last_encounter || 'No encounters yet'}</td>
                <td><span className={getBadgeClass(p.hf_risk)}>{p.hf_risk}% {getBadgeLabel(p.hf_risk)}</span></td>
                <td><span className={getBadgeClass(p.mortality_risk)}>{p.mortality_risk}% {getBadgeLabel(p.mortality_risk)}</span></td>
                <td className={p.status === 'Active' ? 'status-active' : 'status-discharged'}>
                  {p.status === 'Active' ? '● Active' : '○ Discharged'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
