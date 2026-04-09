// src/components/ECGViewer.jsx
import { useEffect, useRef } from 'react';

const LEADS = ['Lead I', 'Lead II', 'Lead III', 'aVR', 'aVL', 'aVF'];

function drawLead(canvas, leadIdx, findings = []) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#050d1a';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(34,197,94,0.07)';
  ctx.lineWidth = 0.5;
  for (let x = 0; x < W; x += 16) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y < H; y += 16) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

  // Waveform
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 1.5;
  ctx.shadowColor = '#22c55e';
  ctx.shadowBlur = 3;
  ctx.beginPath();

  const mid = H * 0.55;
  // Vary waveform per lead for realism
  const afib = findings.some(f => f?.toLowerCase().includes('fibrill'));
  const lbbb = findings.some(f => f?.toLowerCase().includes('bundle'));
  const rAmp = leadIdx === 3 ? -35 : leadIdx === 4 ? -20 : leadIdx === 5 ? 18 : 30 + leadIdx * 3;
  const noise = afib ? () => (Math.random() - 0.5) * 4 : () => 0;
  const beatW = lbbb ? 96 : 80;

  let x = 4;
  ctx.moveTo(x, mid + noise());
  while (x < W - 10) {
    const n = noise();
    ctx.lineTo(x + 10, mid + n);       x += 10; // flat
    ctx.quadraticCurveTo(x+5, mid-7+n, x+10, mid+n); x += 10; // P wave
    ctx.lineTo(x + 6, mid + n);        x += 6;  // PR
    ctx.lineTo(x + 3, mid + 7 + n);    x += 3;  // Q
    ctx.lineTo(x + 2, mid - rAmp + n); x += 2;  // R spike
    ctx.lineTo(x + 3, mid + 9 + n);    x += 3;  // S
    ctx.lineTo(x + 10, mid + n);       x += 10; // ST
    ctx.quadraticCurveTo(x+8, mid-13+n, x+16, mid+n); x += 16; // T
    ctx.lineTo(x + (beatW - 60), mid + noise()); x += (beatW - 60); // TP
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

export default function ECGViewer({ findings = [] }) {
  const refs = useRef([]);

  useEffect(() => {
    refs.current.forEach((canvas, i) => {
      if (canvas) {
        canvas.width = canvas.offsetWidth || 480;
        canvas.height = 52;
        drawLead(canvas, i, findings);
      }
    });
  }, [findings]);

  return (
    <div className="ecg-area">
      {LEADS.map((lead, i) => (
        <div key={lead} className="ecg-lead">
          <div className="ecg-lead-label">{lead}</div>
          <canvas className="ecg-canvas" ref={el => refs.current[i] = el} height={52} style={{ width: '100%' }} />
        </div>
      ))}
      {findings.length > 0 && (
        <div className="ecg-findings">
          <div className="ecg-findings-title">Findings</div>
          <ul>{findings.map((f, i) => <li key={i}>{f}</li>)}</ul>
        </div>
      )}
    </div>
  );
}
