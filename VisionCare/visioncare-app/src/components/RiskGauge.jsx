// src/components/RiskGauge.jsx
// Semicircle speedometer: starts at left (180°), sweeps RIGHT through TOP, ends at right (0°)
// Uses polarToXY for clarity. SVG angles: 0°=right, 90°=down, 180°=left
// The gauge arc spans from 180° (left tip) to 0° (right tip) going COUNTERCLOCKWISE
// which in SVG screen coords (y-down) means sweep-flag=1 (clockwise visually = going UP through top)

function polarToXY(cx, cy, r, angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  return {
    x: cx + r * Math.cos(rad),
    y: cy - r * Math.sin(rad),  // negate: SVG y-down, arc is upper semicircle
  };
}

export default function RiskGauge({ pct = 0, label, size = 140 }) {
  const cx = 60, cy = 56, r = 44;

  // Background track: full semicircle from 180° to 0° going clockwise (sweep=1)
  const bgStart = polarToXY(cx, cy, r, 180); // left tip
  const bgEnd   = polarToXY(cx, cy, r, 0);   // right tip
  const bgPath  = `M ${bgStart.x} ${bgStart.y} A ${r} ${r} 0 1 1 ${bgEnd.x} ${bgEnd.y}`;

  // Foreground: from 180° sweeping clockwise up to the angle for `pct`
  // angle goes from 180° (0%) to 0° (100%) — i.e., angle = 180 - pct*1.8
  const fgAngle = 180 - Math.min(Math.max(pct, 0), 99.9) * 1.8;
  const fgEnd   = polarToXY(cx, cy, r, fgAngle);
  // large-arc-flag = 1 if arc > 180° (i.e., pct > 100%) — never happens, always 0
  // But we need large-arc-flag=1 when swept angle > 180°: swept = 180 - fgAngle
  const sweptDeg = 180 - fgAngle; // = pct * 1.8
  const largeArc = sweptDeg > 180 ? 1 : 0;
  const fgPath = pct > 0
    ? `M ${bgStart.x} ${bgStart.y} A ${r} ${r} 0 ${largeArc} 1 ${fgEnd.x.toFixed(2)} ${fgEnd.y.toFixed(2)}`
    : '';

  const color = pct >= 70 ? '#ef4444' : pct >= 40 ? '#f59e0b' : '#22c55e';

  // Needle line from center to current angle
  const needleTip  = polarToXY(cx, cy, r - 10, fgAngle);
  const needleTail = polarToXY(cx, cy, -8,      fgAngle);

  // Tick marks at 0%, 25%, 50%, 75%, 100%
  const ticks = [0, 25, 50, 75, 100].map(t => {
    const a = 180 - t * 1.8;
    const inner = polarToXY(cx, cy, r - 7, a);
    const outer = polarToXY(cx, cy, r + 2, a);
    return { inner, outer, t };
  });

  return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:2 }}>
      <svg viewBox="0 0 120 68" width={size} height={size * 0.6}
           style={{ overflow:'visible', display:'block' }}>

        {/* Background track */}
        <path d={bgPath} fill="none"
              stroke="rgba(255,255,255,0.08)" strokeWidth="10"
              strokeLinecap="round" />

        {/* Colored fill arc */}
        {fgPath && (
          <path d={fgPath} fill="none"
                stroke={color} strokeWidth="10"
                strokeLinecap="round"
                style={{ filter: `drop-shadow(0 0 6px ${color}99)` }} />
        )}

        {/* Tick marks */}
        {ticks.map(({ inner, outer }, i) => (
          <line key={i}
                x1={inner.x.toFixed(2)} y1={inner.y.toFixed(2)}
                x2={outer.x.toFixed(2)} y2={outer.y.toFixed(2)}
                stroke="rgba(255,255,255,0.20)" strokeWidth="1.5" />
        ))}

        {/* Needle */}
        {pct > 0 && (
          <>
            <line
              x1={needleTail.x.toFixed(2)} y1={needleTail.y.toFixed(2)}
              x2={needleTip.x.toFixed(2)}  y2={needleTip.y.toFixed(2)}
              stroke={color} strokeWidth="2" strokeLinecap="round"
              style={{ filter:`drop-shadow(0 0 3px ${color})` }}
            />
            <circle cx={cx} cy={cy} r="3.5" fill={color}
                    style={{ filter:`drop-shadow(0 0 4px ${color})` }} />
          </>
        )}
      </svg>

      <div style={{ color, fontSize:22, fontWeight:700, letterSpacing:'-0.5px',
                    marginTop:-4, textShadow:`0 0 12px ${color}66` }}>
        {pct}%
      </div>
      <div style={{ color:'rgba(255,255,255,0.5)', fontSize:11,
                    textTransform:'uppercase', letterSpacing:'0.08em' }}>
        {label}
      </div>
    </div>
  );
}
