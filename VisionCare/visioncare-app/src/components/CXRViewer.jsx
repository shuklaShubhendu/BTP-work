// src/components/CXRViewer.jsx
import { useState } from 'react';
import { User } from 'lucide-react';

export default function CXRViewer({ imageSrc = null, findings = [], severity = 'normal' }) {
  const [gradcam, setGradcam] = useState(false);

  const gradcamStyle = {
    background: severity === 'critical'
      ? 'radial-gradient(ellipse at 45% 55%, rgba(239,68,68,0.45) 0%, rgba(245,158,11,0.25) 35%, transparent 65%)'
      : severity === 'moderate'
      ? 'radial-gradient(ellipse at 48% 52%, rgba(245,158,11,0.35) 0%, rgba(234,179,8,0.15) 40%, transparent 65%)'
      : 'radial-gradient(ellipse at 50% 50%, rgba(34,197,94,0.2) 0%, transparent 60%)',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
      <div className="cxr-area">
        {imageSrc ? (
          <>
            <img src={imageSrc} alt="Chest X-Ray" className="cxr-image" style={{ filter: 'grayscale(0.3) contrast(1.1)' }} />
            {gradcam && (
              <div className="gradcam-overlay" style={{ ...gradcamStyle, mixBlendMode: 'screen', opacity: 0.85 }} />
            )}
          </>
        ) : (
          <div className="cxr-placeholder">
            <div style={{ width: 90, height: 110, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
              <User size={40} color="rgba(255,255,255,0.15)" />
              {gradcam && (
                <div style={{ ...gradcamStyle, position: 'absolute', inset: 0, borderRadius: 10, mixBlendMode: 'screen' }} />
              )}
            </div>
            <span>Chest X-Ray</span>
            <small>PA View · 224×224</small>
            {!imageSrc && <small style={{ color: 'var(--amber)', fontSize: 10 }}>📁 Add image to demo_images/</small>}
          </div>
        )}
      </div>

      {/* Grad-CAM toggle */}
      <div className="gradcam-toggle-bar">
        <label className="toggle" style={{ cursor: 'pointer' }}>
          <input type="checkbox" checked={gradcam} onChange={e => setGradcam(e.target.checked)} />
          <div className="toggle-track" />
          <div className="toggle-thumb" style={{ transform: gradcam ? 'translateX(14px)' : 'none' }} />
        </label>
        <span>Grad-CAM Overlay</span>
        {gradcam && <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--amber)' }}>
          {severity === 'critical' ? '🔴 High activation in cardiac region' : severity === 'moderate' ? '🟡 Moderate cardiomegaly region' : '🟢 No significant activation'}
        </span>}
      </div>

      {/* Findings chips */}
      {findings.length > 0 && (
        <div style={{ padding: '10px 14px', display: 'flex', flexWrap: 'wrap', gap: 6, background: 'rgba(0,0,0,0.2)' }}>
          {findings.map((f, i) => (
            <span key={i} style={{ padding: '3px 10px', background: 'rgba(34,197,94,0.08)', color: 'var(--text-green)', border: '1px solid rgba(34,197,94,0.15)', borderRadius: 20, fontSize: 11 }}>{f}</span>
          ))}
        </div>
      )}
    </div>
  );
}
