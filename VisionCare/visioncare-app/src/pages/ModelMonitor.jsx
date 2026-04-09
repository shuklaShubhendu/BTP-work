// src/pages/ModelMonitor.jsx
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, BarChart, Bar, Cell } from 'recharts';

const TRAINING_CURVE = [
  {epoch:'A1',v2_auc:0.741,v3_auc:0.732},{epoch:'A2',v2_auc:0.758,v3_auc:0.751},
  {epoch:'A3',v2_auc:0.763,v3_auc:0.765},{epoch:'A4',v2_auc:0.769,v3_auc:0.771},
  {epoch:'A5',v2_auc:0.773,v3_auc:0.776},{epoch:'B1',v2_auc:0.775,v3_auc:0.779},
  {epoch:'B5',v2_auc:0.783,v3_auc:0.782},{epoch:'B10',v2_auc:0.798,v3_auc:0.788},
  {epoch:'B15',v2_auc:0.806,v3_auc:0.792},{epoch:'B20',v2_auc:0.810,v3_auc:0.790},
];

const RADAR_DATA = [
  {metric:'HF AUC',      v2:0.82, v3:0.75},
  {metric:'Mort AUC',    v2:0.80, v3:0.83},
  {metric:'F1 Score',    v2:0.65, v3:0.67},
  {metric:'Recall',      v2:0.72, v3:0.74},
  {metric:'Precision',   v2:0.61, v3:0.62},
  {metric:'Calibration', v2:0.78, v3:0.71},
];

const CONF_MATRIX = [
  { label:'TP', val:482, color:'#22c55e' },
  { label:'FP', val:87,  color:'#f59e0b' },
  { label:'FN', val:115, color:'#f59e0b' },
  { label:'TN', val:1066,color:'#3b82f6' },
];

const TOOLTIP_STYLE = { background:'#1e293b', border:'1px solid rgba(255,255,255,0.1)', borderRadius:8, color:'#f1f5f9', fontSize:12 };

export default function ModelMonitor() {
  return (
    <div className="page-content section-gap">

      {/* Header stats */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(5,1fr)', gap:14 }}>
        {[
          ['MACRO AUC',    '0.8105', 'var(--green)',         'V2 Active'],
          ['HF AUC',       '0.8189', '#60a5fa',             'Heart Failure'],
          ['MORT AUC',     '0.8022', '#a78bfa',             'Mortality'],
          ['TRAINING SET', '10,000', 'var(--text-primary)', 'SYMILE-MIMIC admissions'],
          ['LATENCY',      '1.4s',   'var(--amber)',         'Avg inference time'],
        ].map(([label, val, color, sub]) => (
          <div key={label} className="stat-card">
            <span className="stat-label">{label}</span>
            <div className="stat-val" style={{ fontSize:22, color: color||'inherit' }}>{val}</div>
            <div className="stat-delta">{sub}</div>
          </div>
        ))}
      </div>

      {/* Training curves + Radar */}
      <div style={{ display:'grid', gridTemplateColumns:'1.5fr 1fr', gap:14 }}>
        <div className="card">
          <div className="card-title">Training AUC Curve — Phase A (Frozen) → Phase B (Unfrozen)</div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={TRAINING_CURVE}>
              <defs>
                <linearGradient id="v2grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"   stopColor="#22c55e" stopOpacity={0.3} />
                  <stop offset="95%"  stopColor="#22c55e" stopOpacity={0}   />
                </linearGradient>
                <linearGradient id="v3grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}   />
                </linearGradient>
              </defs>
              <XAxis dataKey="epoch" tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0.7, 0.85]} tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(2)} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => v.toFixed(4)} />
              <Area type="monotone" dataKey="v2_auc" stroke="#22c55e" fill="url(#v2grad)" strokeWidth={2} name="V2 (Active)" />
              <Area type="monotone" dataKey="v3_auc" stroke="#3b82f6" fill="url(#v3grad)" strokeWidth={2} name="V3 (SYMILE)" strokeDasharray="4 2" />
            </AreaChart>
          </ResponsiveContainer>
          <div style={{ display:'flex', gap:16, marginTop:8 }}>
            <span style={{ fontSize:11, color:'#22c55e' }}>— V2 (Macro AUC 0.8105)</span>
            <span style={{ fontSize:11, color:'#3b82f6' }}>- - V3 (Training)</span>
            <span style={{ fontSize:11, color:'var(--text-muted)' }}>|B| = Phase B unfroze encoders</span>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Multi-Metric Radar — V2 vs V3</div>
          <ResponsiveContainer width="100%" height={200}>
            <RadarChart data={RADAR_DATA}>
              <PolarGrid stroke="rgba(255,255,255,0.06)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill:'#64748b', fontSize:10 }} />
              <Radar name="V2" dataKey="v2" stroke="#22c55e" fill="#22c55e" fillOpacity={0.15} strokeWidth={2} />
              <Radar name="V3" dataKey="v3" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1}  strokeWidth={2} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion matrix + Perf bars */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
        <div className="card">
          <div className="card-title">Confusion Matrix — HF Task (Val Set)</div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginTop:8 }}>
            {CONF_MATRIX.map(c => (
              <div key={c.label} style={{ background:'var(--bg-elevated)', border:`1px solid ${c.color}33`, borderRadius:10, padding:'20px', textAlign:'center' }}>
                <div style={{ fontSize:10, color:'var(--text-muted)', marginBottom:6, fontWeight:600, letterSpacing:'0.05em' }}>{c.label}</div>
                <div style={{ fontSize:32, fontWeight:900, color:c.color }}>{c.val}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop:12, display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:8 }}>
            {[['Sensitivity', `${(482/(482+115)*100).toFixed(1)}%`, '#22c55e'], ['Specificity', `${(1066/(1066+87)*100).toFixed(1)}%`, '#3b82f6'], ['PPV', `${(482/(482+87)*100).toFixed(1)}%`, '#a78bfa']].map(([k,v,c]) => (
              <div key={k} style={{ background:'var(--bg-elevated)', borderRadius:8, padding:'10px', textAlign:'center' }}>
                <div style={{ fontSize:10, color:'var(--text-muted)', marginBottom:4 }}>{k}</div>
                <div style={{ fontSize:18, fontWeight:800, color:c }}>{v}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-title">Per-Task AUC — V2 vs V3</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={[
              { task:'Mortality', v2:0.8022, v3:0.8289 },
              { task:'Heart Failure', v2:0.8189, v3:0.7519 },
              { task:'Macro Avg', v2:0.8105, v3:0.7904 },
            ]} barCategoryGap="25%">
              <XAxis dataKey="task" tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0.7,0.88]} tick={{ fill:'#64748b', fontSize:10 }} axisLine={false} tickLine={false} tickFormatter={v=>v.toFixed(2)} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => v.toFixed(4)} />
              <Bar dataKey="v2" fill="#22c55e" radius={[4,4,0,0]} name="V2 Active" />
              <Bar dataKey="v3" fill="#3b82f6" radius={[4,4,0,0]} name="V3 SYMILE" />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ display:'flex', gap:16, marginTop:4 }}>
            <span style={{ fontSize:11, color:'#22c55e' }}>■ V2 (Current Production)</span>
            <span style={{ fontSize:11, color:'#3b82f6' }}>■ V3 (Experimental)</span>
          </div>

          {/* Architecture cards */}
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10, marginTop:16 }}>
            {[{ label:'V2 Active', color:'#22c55e', enc:['ConvNeXt-Tiny (CXR)','1D-CNN (ECG)','2-layer MLP (Labs)'], auc:'0.8105' },
              { label:'V3 Exp.',  color:'#3b82f6', enc:['ResNet-50/SYMILE (CXR)','1D-ResNet18 (ECG)','3-layer NN (Labs)'], auc:'0.7904' }
            ].map(({ label, color, enc, auc }) => (
              <div key={label} style={{ background:'var(--bg-elevated)', border:`1px solid ${color}33`, borderRadius:8, padding:'12px' }}>
                <div style={{ fontSize:12, fontWeight:700, color, marginBottom:8 }}>{label}</div>
                {enc.map(e => <div key={e} style={{ fontSize:11, color:'var(--text-secondary)', marginBottom:3 }}>• {e}</div>)}
                <div style={{ marginTop:8, fontSize:13, fontWeight:800, color }}>AUC {auc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model swap info */}
      <div className="card" style={{ background:'rgba(34,197,94,0.04)', borderColor:'rgba(34,197,94,0.2)' }}>
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
          <div>
            <div style={{ fontSize:14, fontWeight:700 }}>🔄 Model Version Control</div>
            <div style={{ fontSize:12, color:'var(--text-secondary)', marginTop:4 }}>Currently serving <strong style={{ color:'var(--green)' }}>V2 (AUC 0.8105)</strong>. To swap to V3, update <code style={{ background:'var(--bg-elevated)', padding:'2px 6px', borderRadius:4, fontSize:11 }}>MODEL_VERSION = "V3"</code> in <code style={{ background:'var(--bg-elevated)', padding:'2px 6px', borderRadius:4, fontSize:11 }}>backend/main.py</code>.</div>
          </div>
          <div style={{ background:'var(--green-dim)', border:'1px solid var(--green-border)', borderRadius:8, padding:'8px 16px', fontSize:12, color:'var(--green)', fontWeight:700, whiteSpace:'nowrap' }}>
            ✓ V2 ACTIVE
          </div>
        </div>
      </div>
    </div>
  );
}
