// src/pages/ModelMonitor.jsx — VisionCare 3.0 Model Monitor
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, BarChart, Bar, Cell } from 'recharts';

const TRAINING_CURVE = [
  {epoch:'A1',auc:0.732},{epoch:'A2',auc:0.751},{epoch:'A3',auc:0.765},
  {epoch:'A4',auc:0.771},{epoch:'A5',auc:0.776},
  {epoch:'B1',auc:0.769},{epoch:'B3',auc:0.775},{epoch:'B5',auc:0.779},
  {epoch:'B8',auc:0.784},{epoch:'B10',auc:0.788},{epoch:'B12',auc:0.791},
  {epoch:'B15',auc:0.793},{epoch:'B17',auc:0.7926},{epoch:'B20',auc:0.790},
];

const GATE_EVOLUTION = [
  {epoch:'A1',vision:0.55,signal:0.20,clinical:0.25},
  {epoch:'A3',vision:0.48,signal:0.28,clinical:0.24},
  {epoch:'A5',vision:0.42,signal:0.36,clinical:0.22},
  {epoch:'B1',vision:0.40,signal:0.35,clinical:0.25},
  {epoch:'B5',vision:0.38,signal:0.34,clinical:0.28},
  {epoch:'B10',vision:0.36,signal:0.34,clinical:0.30},
  {epoch:'B15',vision:0.34,signal:0.34,clinical:0.32},
  {epoch:'B17',vision:0.339,signal:0.338,clinical:0.323},
  {epoch:'B20',vision:0.34,signal:0.34,clinical:0.32},
];

const PER_CLASS_AUC = [
  { target:'Sepsis',   auc:0.8223 },
  { target:'PE',       auc:0.8213 },
  { target:'Mortality',auc:0.8082 },
  { target:'Arrhyth.', auc:0.7936 },
  { target:'ICU',      auc:0.7839 },
  { target:'HF',       auc:0.7829 },
  { target:'AKI',      auc:0.7813 },
  { target:'MI',       auc:0.7473 },
];

const RADAR_DATA = [
  {metric:'Mortality',  v2p2:0.800, v3:0.808},
  {metric:'HF',         v2p2:0.738, v3:0.783},
  {metric:'MI',         v2p2:0.687, v3:0.747},
  {metric:'Arrhythmia', v2p2:0.789, v3:0.794},
  {metric:'Sepsis',     v2p2:0.788, v3:0.822},
  {metric:'PE',         v2p2:0.798, v3:0.821},
  {metric:'AKI',        v2p2:0.751, v3:0.781},
  {metric:'ICU',        v2p2:0.787, v3:0.784},
];

const TOOLTIP_STYLE = { background:'#1e293b', border:'1px solid rgba(255,255,255,0.1)', borderRadius:8, color:'#f1f5f9', fontSize:12 };

export default function ModelMonitor() {
  return (
    <div className="page-content section-gap">

      {/* Header stats */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(5,1fr)', gap:14 }}>
        {[
          ['MACRO AUC',    '0.7926', 'var(--green)',         'V3 Active (8 diseases)'],
          ['BEST AUC',     '0.8223', '#60a5fa',             'Sepsis (highest)'],
          ['GATE BALANCE', '34/34/32', '#a78bfa',           'V:S:C balanced'],
          ['TRAINING SET', '10,000', 'var(--text-primary)', 'SYMILE-MIMIC admissions'],
          ['BEST EPOCH',   '17/25',  'var(--amber)',         'Phase A(5) + Phase B(12)'],
        ].map(([label, val, color, sub]) => (
          <div key={label} className="stat-card">
            <span className="stat-label">{label}</span>
            <div className="stat-val" style={{ fontSize:22, color: color||'inherit' }}>{val}</div>
            <div className="stat-delta">{sub}</div>
          </div>
        ))}
      </div>

      {/* Training curve + Gate Evolution */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
        <div className="card">
          <div className="card-title">Training AUC — Phase A (Frozen) → Phase B (Unfrozen)</div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={TRAINING_CURVE}>
              <defs>
                <linearGradient id="aucgrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"   stopColor="#22c55e" stopOpacity={0.3} />
                  <stop offset="95%"  stopColor="#22c55e" stopOpacity={0}   />
                </linearGradient>
              </defs>
              <XAxis dataKey="epoch" tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0.72, 0.80]} tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(2)} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => v.toFixed(4)} />
              <Area type="monotone" dataKey="auc" stroke="#22c55e" fill="url(#aucgrad)" strokeWidth={2} name="Macro AUC" />
            </AreaChart>
          </ResponsiveContainer>
          <div style={{ display:'flex', gap:16, marginTop:8 }}>
            <span style={{ fontSize:11, color:'#22c55e' }}>— V3 Macro AUC (0.7926)</span>
            <span style={{ fontSize:11, color:'var(--text-muted)' }}>|B| = Phase B unfroze Layer3+4</span>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Gate Evolution — Vision / Signal / Clinical</div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={GATE_EVOLUTION}>
              <defs>
                <linearGradient id="vGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="sGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22c55e" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="cGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#a78bfa" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="epoch" tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0.15, 0.6]} tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} tickFormatter={v => (v*100).toFixed(0)+'%'} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => (v*100).toFixed(1)+'%'} />
              <Area type="monotone" dataKey="vision" stroke="#38bdf8" fill="url(#vGrad)" strokeWidth={2} name="Vision (CXR)" />
              <Area type="monotone" dataKey="signal" stroke="#22c55e" fill="url(#sGrad)" strokeWidth={2} name="Signal (ECG)" />
              <Area type="monotone" dataKey="clinical" stroke="#a78bfa" fill="url(#cGrad)" strokeWidth={2} name="Clinical (Labs)" />
            </AreaChart>
          </ResponsiveContainer>
          <div style={{ display:'flex', gap:16, marginTop:8 }}>
            <span style={{ fontSize:11, color:'#38bdf8' }}>— Vision 33.9%</span>
            <span style={{ fontSize:11, color:'#22c55e' }}>— Signal 33.8%</span>
            <span style={{ fontSize:11, color:'#a78bfa' }}>— Clinical 32.3%</span>
          </div>
        </div>
      </div>

      {/* Per-disease AUC + Radar comparison */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
        <div className="card">
          <div className="card-title">Per-Disease AUC — All 8 Targets</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={PER_CLASS_AUC} layout="vertical" barCategoryGap="15%">
              <XAxis type="number" domain={[0.7, 0.85]} tick={{ fill:'#64748b', fontSize:10 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(2)} />
              <YAxis type="category" dataKey="target" tick={{ fill:'#94a3b8', fontSize:11 }} axisLine={false} tickLine={false} width={70} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => v.toFixed(4)} />
              <Bar dataKey="auc" radius={[0,4,4,0]} name="AUC-ROC">
                {PER_CLASS_AUC.map((entry, i) => (
                  <Cell key={i} fill={entry.auc >= 0.80 ? '#22c55e' : entry.auc >= 0.78 ? '#3b82f6' : '#f59e0b'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">V2 Phase 2 vs V3 — Radar Comparison</div>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={RADAR_DATA}>
              <PolarGrid stroke="rgba(255,255,255,0.06)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill:'#64748b', fontSize:10 }} />
              <Radar name="V2 Phase 2" dataKey="v2p2" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.1} strokeWidth={2} />
              <Radar name="V3" dataKey="v3" stroke="#22c55e" fill="#22c55e" fillOpacity={0.15} strokeWidth={2} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
            </RadarChart>
          </ResponsiveContainer>
          <div style={{ display:'flex', gap:16, marginTop:8, justifyContent:'center' }}>
            <span style={{ fontSize:11, color:'#f59e0b' }}>— V2 Phase 2 (AUC 0.7672)</span>
            <span style={{ fontSize:11, color:'#22c55e' }}>— V3 (AUC 0.7926)</span>
          </div>
        </div>
      </div>

      {/* Architecture cards */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
        <div className="card" style={{ borderColor:'rgba(249,168,38,0.15)' }}>
          <div style={{ fontSize:14, fontWeight:700, color:'#f59e0b', marginBottom:12 }}>V2 Phase 2 — Cross-Attention Fusion</div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12 }}>
            {[['Encoders','ConvNeXt-Tiny (CXR)\n1D-CNN (ECG)\n2-layer MLP (Labs)'],
              ['Fusion','Cross-Attention Gating\n4-Head, Frozen Encoders'],
              ['Targets','8 (same diseases)'],
              ['AUC','0.7672 Macro'],
              ['Gates','V=79% S=15% C=6%\n(Imbalanced!)'],
              ['Loss','BCE + Label Smooth']
            ].map(([k,v]) => (
              <div key={k} style={{ background:'var(--bg-elevated)', borderRadius:8, padding:'10px 12px' }}>
                <div style={{ fontSize:10, color:'var(--text-muted)', marginBottom:4, fontWeight:600 }}>{k}</div>
                <div style={{ fontSize:11, color:'var(--text-secondary)', whiteSpace:'pre-line' }}>{v}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="card" style={{ borderColor:'rgba(34,197,94,0.2)' }}>
          <div style={{ fontSize:14, fontWeight:700, color:'#22c55e', marginBottom:12 }}>V3 — Progressive Unfreeze Fusion</div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12 }}>
            {[['Encoders','ResNet-50 (CXR, 2048-D)\n1D ResNet-18 (ECG, 512-D)\n3-Layer NN (Labs, 256-D)'],
              ['Fusion','Cross-Attention + FFN\n4-Head + Gate Entropy Reg'],
              ['Targets','8 diseases'],
              ['AUC','0.7926 Macro (+2.5%)'],
              ['Gates','V=34% S=34% C=32%\nBalanced!'],
              ['Loss','Focal (γ=2) + EMA']
            ].map(([k,v]) => (
              <div key={k} style={{ background:'var(--bg-elevated)', borderRadius:8, padding:'10px 12px' }}>
                <div style={{ fontSize:10, color:'var(--text-muted)', marginBottom:4, fontWeight:600 }}>{k}</div>
                <div style={{ fontSize:11, color:'var(--text-secondary)', whiteSpace:'pre-line' }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model status */}
      <div className="card" style={{ background:'rgba(34,197,94,0.04)', borderColor:'rgba(34,197,94,0.2)' }}>
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
          <div>
            <div style={{ fontSize:14, fontWeight:700 }}>Model Version Control</div>
            <div style={{ fontSize:12, color:'var(--text-secondary)', marginTop:4 }}>Currently serving <strong style={{ color:'var(--green)' }}>V3 (Macro AUC 0.7926, 8 diseases, balanced gates 34/34/32)</strong>. Checkpoint: <code style={{ background:'var(--bg-elevated)', padding:'2px 6px', borderRadius:4, fontSize:11 }}>fusion_v3_best.pth</code></div>
          </div>
          <div style={{ background:'var(--green-dim)', border:'1px solid var(--green-border)', borderRadius:8, padding:'8px 16px', fontSize:12, color:'var(--green)', fontWeight:700, whiteSpace:'nowrap' }}>
            ✓ V3 ACTIVE
          </div>
        </div>
      </div>
    </div>
  );
}
