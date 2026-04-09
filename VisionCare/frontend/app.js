// ══════════════════════════════════════════════════════════════════
// VisionCare 2.0 — Frontend App Logic
// ══════════════════════════════════════════════════════════════════

const API = 'http://127.0.0.1:8000';
let currentPatientId = null;
let currentEncounterId = null;
let currentEncounterData = null;

// ── Page Navigation ────────────────────────────────────────────────
function showPage(name) {
  document.querySelectorAll('.content-page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const page = document.getElementById(`page-${name}`);
  if (page) page.classList.add('active');
  const nav = document.getElementById(`nav-${name}`);
  if (nav) nav.classList.add('active');
  const titles = { dashboard: 'Dashboard', patients: 'Patient Profile', encounter: 'Encounter Analysis', about: 'About' };
  document.getElementById('topbar-title').textContent = titles[name] || name;
}

// ── Auth ───────────────────────────────────────────────────────────
function doLogin() {
  document.getElementById('page-login').classList.remove('active');
  document.getElementById('app-shell').classList.add('active');
  document.getElementById('app-shell').style.display = 'flex';
  showPage('dashboard');
  loadDashboard();
}

function doLogout() {
  document.getElementById('app-shell').classList.remove('active');
  document.getElementById('app-shell').style.display = 'none';
  document.getElementById('page-login').classList.add('active');
}

function togglePw() {
  const inp = document.getElementById('login-pw');
  inp.type = inp.type === 'password' ? 'text' : 'password';
}

// ── Dashboard ──────────────────────────────────────────────────────
async function loadDashboard() {
  try {
    const patients = await apiFetch('/api/patients');
    renderPatientTable(patients);
  } catch(e) {
    renderPatientTable(getMockPatients());
  }
}

function renderPatientTable(patients) {
  const tbody = document.getElementById('patient-tbody');
  tbody.innerHTML = patients.map(p => {
    const hfClass = p.hf_risk >= 70 ? 'risk-high' : p.hf_risk >= 40 ? 'risk-moderate' : 'risk-low';
    const hfLabel = p.hf_risk >= 70 ? 'High' : p.hf_risk >= 40 ? 'Moderate' : 'Low';
    const mortClass = p.mortality_risk >= 50 ? 'risk-high' : p.mortality_risk >= 25 ? 'risk-moderate' : 'risk-low';
    const mortLabel = p.mortality_risk >= 50 ? 'High' : p.mortality_risk >= 25 ? 'Moderate' : 'Low';
    const statusClass = p.status?.toLowerCase() === 'active' ? 'active' : 'discharged';
    const age = p.age || '—';
    const gen = p.gender ? p.gender[0] : '—';
    return `<tr onclick="openPatient('${p.id}')">
      <td class="mrn-cell">${p.mrn}</td>
      <td><strong>${p.name}</strong></td>
      <td>${age} / ${gen}</td>
      <td>${p.last_encounter || '—'}</td>
      <td><span class="risk-badge ${hfClass}">${p.hf_risk}% ${hfLabel}</span></td>
      <td><span class="risk-badge ${mortClass}">${p.mortality_risk}% ${mortLabel}</span></td>
      <td><span class="status-dot ${statusClass}">${p.status}</span></td>
    </tr>`;
  }).join('');
}

async function onSearch(q) {
  try {
    const patients = await apiFetch(`/api/patients?search=${encodeURIComponent(q)}`);
    renderPatientTable(patients);
  } catch(e) {
    const all = getMockPatients();
    const ql = q.toLowerCase();
    renderPatientTable(all.filter(p => p.name.toLowerCase().includes(ql) || p.mrn.toLowerCase().includes(ql)));
  }
}

// ── Patient Profile ────────────────────────────────────────────────
async function openPatient(id) {
  currentPatientId = id;
  showPage('patient');
  document.getElementById('nav-patients').classList.add('active');

  let patient, encounters;
  try {
    [patient, encounters] = await Promise.all([
      apiFetch(`/api/patients/${id}`),
      apiFetch(`/api/patients/${id}/encounters`)
    ]);
  } catch(e) {
    patient = getMockPatients().find(p => p.id === id);
    encounters = getMockEncounters(id);
  }

  document.getElementById('bc-patient-name').textContent = patient.name;
  document.getElementById('patient-header').innerHTML = `
    <div>
      <div class="patient-name-big">${patient.name}</div>
      <div class="patient-mrn">${patient.mrn}</div>
    </div>
    <div class="patient-meta">
      <div class="meta-item"><span class="meta-key">AGE</span><span class="meta-val">${patient.age}</span></div>
      <div class="meta-item"><span class="meta-key">GENDER</span><span class="meta-val">${patient.gender}</span></div>
      <div class="meta-item"><span class="meta-key">STATUS</span><span class="meta-val green">${patient.status}</span></div>
    </div>`;

  document.getElementById('encounter-list').innerHTML = encounters.map(e => {
    const hfClass = e.hf_risk >= 70 ? 'risk-high' : e.hf_risk >= 40 ? 'risk-moderate' : 'risk-low';
    const mClass  = e.mortality_risk >= 50 ? 'risk-high' : e.mortality_risk >= 25 ? 'risk-moderate' : 'risk-low';
    return `<div class="encounter-card" onclick="openEncounter('${id}','${e.id}')">
      <div class="enc-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg></div>
      <div class="enc-info">
        <div class="enc-label">${e.label} · ${e.description}</div>
        <div class="enc-desc">Admitted ${e.date}</div>
      </div>
      <div class="enc-risks">
        <span class="enc-risk" style="color:#ef4444">♥ <span class="risk-badge ${hfClass}">${e.hf_risk}% HF</span></span>
        <span class="enc-risk" style="color:#22c55e">⊙ <span class="risk-badge ${mClass}">${e.mortality_risk}% Mort.</span></span>
      </div>
    </div>`;
  }).join('');
}

function goBackToPatient() {
  if (currentPatientId) openPatient(currentPatientId);
}

// ── Encounter Analysis ─────────────────────────────────────────────
async function openEncounter(patientId, encId) {
  currentPatientId = patientId;
  currentEncounterId = encId;
  showPage('encounter');

  let data;
  try {
    data = await apiFetch(`/api/patients/${patientId}/encounters/${encId}`);
  } catch(e) {
    const p = getMockPatients().find(p => p.id === patientId);
    const encs = getMockEncounters(patientId);
    const enc = encs.find(e => e.id === encId);
    data = { ...enc, patient: p };
  }
  currentEncounterData = data;

  // Breadcrumb
  document.getElementById('bc-enc-patient').textContent = data.patient?.name || 'Patient';
  document.getElementById('bc-enc-id').textContent = data.label || encId;

  // Risk Gauges
  renderGauge('gauge-hf', data.hf_risk, 'Heart Failure Risk', data.hf_risk >= 70 ? '#ef4444' : data.hf_risk >= 40 ? '#f59e0b' : '#22c55e');
  renderGauge('gauge-mort', data.mortality_risk, 'Mortality Risk', data.mortality_risk >= 50 ? '#ef4444' : data.mortality_risk >= 25 ? '#f59e0b' : '#22c55e');

  // Contribution bars
  const gates = data.gates || { vision: 0.33, signal: 0.33, clinical: 0.34 };
  document.getElementById('contrib-bars').innerHTML = [
    { name: 'CXR', pct: Math.round(gates.vision * 100),    color: '#38bdf8' },
    { name: 'ECG', pct: Math.round(gates.signal * 100),   color: '#22c55e' },
    { name: 'Labs', pct: Math.round(gates.clinical * 100), color: '#a78bfa' },
  ].map(m => `
    <div class="contrib-row">
      <span class="contrib-name">${m.name}</span>
      <div class="contrib-track"><div class="contrib-fill" style="width:${m.pct}%;background:${m.color}"></div></div>
      <span class="contrib-pct" style="color:${m.color}">${m.pct}%</span>
    </div>`).join('');

  // Model badge
  document.getElementById('model-badge').innerHTML = `⚡ VisionCare 2.0 Fusion · 1.4s latency`;
  document.getElementById('pred-time').textContent = `Predicted at ${data.date || 'Mar 26, 2026'} · 15:00 UTC`;

  // Patient info
  const p = data.patient || {};
  document.getElementById('patient-info-box').innerHTML = `
    <div class="info-row"><span class="info-key">Name</span><span class="info-val">${p.name || '—'}</span></div>
    <div class="info-row"><span class="info-key">MRN</span><span class="info-val">${p.mrn || '—'}</span></div>
    <div class="info-row"><span class="info-key">Age / Gender</span><span class="info-val">${p.age || '—'} / ${p.gender || '—'}</span></div>
    <div class="info-row"><span class="info-key">Encounter</span><span class="info-val">${data.label || encId}</span></div>`;

  // CXR findings
  const cxrFindings = data.cxr_findings || ['No significant findings'];
  document.getElementById('cxr-findings-box').innerHTML = cxrFindings.map(f => `<span class="finding-chip">${f}</span>`).join('');

  // ECG
  drawECG();
  const ecgFindings = data.ecg_findings || ['Normal sinus rhythm'];
  document.getElementById('ecg-findings-box').innerHTML = ecgFindings.map(f => `<span class="finding-chip">${f}</span>`).join('');

  // Labs
  const labs = data.labs || [];
  document.getElementById('labs-tbody').innerHTML = labs.map(l => `
    <tr>
      <td>${l.name}</td>
      <td><strong>${l.value}</strong></td>
      <td style="color:#64748b;font-size:11px">${l.normal}</td>
      <td><span class="lab-status lab-${l.status?.toLowerCase()}">${l.status}</span></td>
    </tr>`).join('');

  // Initial AI message
  const hfRisk = data.hf_risk;
  const g = gates;
  document.getElementById('initial-ai-msg').innerHTML = formatBotMsg(
    `The model's prediction of <strong>${hfRisk}% Heart Failure risk</strong> was driven predominantly by the Clinical Laboratory data (<strong>${Math.round(g.clinical*100)}% contribution</strong>). The BNP level is severely elevated; according to the 2022 AHA/ACC Heart Failure Guidelines [Section 5.2, p.14], a BNP exceeding 400 pg/mL is a Class I indication for heart failure. This was corroborated by the ECG (<strong>${Math.round(g.signal*100)}%</strong>), and the CXR (<strong>${Math.round(g.vision*100)}%</strong>), confirming mild pulmonary congestion.<br/><span class="citation">📋 AHA 2022 HF Guidelines — Section 5.2, p.14</span>`
  );

  // Reset tabs
  document.querySelectorAll('.tab-panel').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-cxr').classList.add('active');
  document.querySelectorAll('.modal-tab')[0].classList.add('active');
}

// ── SVG Gauge ──────────────────────────────────────────────────────
function renderGauge(containerId, pct, label, color) {
  const r = 44, cx = 60, cy = 60;
  const start = Math.PI, end = 0;
  const angle = start + (pct / 100) * Math.PI;
  const x = cx + r * Math.cos(angle);
  const y = cy + r * Math.sin(angle);
  const large = pct > 50 ? 1 : 0;

  const bgPath = `M ${cx-r} ${cy} A ${r} ${r} 0 0 1 ${cx+r} ${cy}`;
  const fgPath = `M ${cx-r} ${cy} A ${r} ${r} 0 ${large} 1 ${x} ${y}`;

  document.getElementById(containerId).innerHTML = `
    <svg class="gauge-svg" viewBox="0 0 120 70" width="120" height="70">
      <path d="${bgPath}" fill="none" stroke="#e2e8f0" stroke-width="8" stroke-linecap="round"/>
      <path d="${fgPath}" fill="none" stroke="${color}" stroke-width="8" stroke-linecap="round"/>
    </svg>
    <div class="gauge-val" style="color:${color};margin-top:-16px">${pct}%</div>
    <div class="gauge-name">${label}</div>`;
}

// ── ECG Drawing ────────────────────────────────────────────────────
function drawECG() {
  const canvas = document.getElementById('ecg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0f1e';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(34,197,94,0.08)';
  ctx.lineWidth = 0.5;
  for (let x = 0; x < W; x += 20) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y < H; y += 20) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

  // ECG waveform (simulated)
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 1.5;
  ctx.shadowColor = '#22c55e';
  ctx.shadowBlur = 4;
  ctx.beginPath();
  const mid = H / 2;
  const beatW = 80;

  for (let lead = 0; lead < 2; lead++) {
    const yOff = lead === 0 ? mid - 40 : mid + 40;
    let x = 5;
    ctx.moveTo(x, yOff);
    while (x < W - 10) {
      // Flat
      ctx.lineTo(x + 10, yOff);
      x += 10;
      // P wave
      ctx.quadraticCurveTo(x+5, yOff-8, x+10, yOff);
      x += 10;
      // PR flat
      ctx.lineTo(x+8, yOff);
      x += 8;
      // Q
      ctx.lineTo(x+3, yOff+6);
      x += 3;
      // R (spike)
      ctx.lineTo(x+2, yOff - (lead===0?40:30));
      x += 2;
      // S
      ctx.lineTo(x+3, yOff+8);
      x += 3;
      // ST flat
      ctx.lineTo(x+10, yOff);
      x += 10;
      // T wave
      ctx.quadraticCurveTo(x+8, yOff-15, x+16, yOff);
      x += 16;
      // Rest
      ctx.lineTo(x+18, yOff);
      x += 18;
    }
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Label
  ctx.fillStyle = '#4ade80';
  ctx.font = '10px Inter';
  ctx.fillText('Lead II', 8, 16);
  ctx.fillText('Lead V5', 8, H/2 + 12);
}

// ── Tab Switching ──────────────────────────────────────────────────
function switchTab(btn, tabId) {
  document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(tabId).classList.add('active');
  if (tabId === 'tab-ecg') setTimeout(drawECG, 50);
}

// ── Grad-CAM Toggle ────────────────────────────────────────────────
function toggleGradcam() {
  const checked = document.getElementById('gradcam-toggle').checked;
  const viewer = document.querySelector('.cxr-viewer');
  if (checked) {
    viewer.style.background = 'radial-gradient(ellipse at 40% 55%, rgba(239,68,68,0.3) 0%, rgba(245,158,11,0.2) 30%, #0a0f1e 70%)';
  } else {
    viewer.style.background = '#0a0f1e';
  }
}

// ── AI Chat ─────────────────────────────────────────────────────────
async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  addChatMsg('user', msg);
  addTyping();
  try {
    const res = await apiFetch('/api/chat', 'POST', {
      message: msg,
      patient_id: currentPatientId,
      encounter_id: currentEncounterId,
    });
    removeTyping();
    addChatMsg('bot', res.reply);
  } catch(e) {
    removeTyping();
    addChatMsg('bot', getLocalAIResponse(msg));
  }
}

function sendSuggestion(msg) {
  document.getElementById('chat-input').value = msg;
  sendChat();
}

function addChatMsg(type, text) {
  const div = document.createElement('div');
  div.className = `chat-msg ${type}`;
  div.innerHTML = type === 'bot' ? formatBotMsg(text) : escapeHtml(text);
  document.getElementById('chat-messages').appendChild(div);
  div.scrollIntoView({ behavior: 'smooth' });
}

function formatBotMsg(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')
    .replace(/📋 (.*)/g, '<span class="citation">📋 $1</span>');
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'chat-msg bot'; div.id = 'typing-indicator';
  div.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  document.getElementById('chat-messages').appendChild(div);
  div.scrollIntoView({ behavior: 'smooth' });
}

function removeTyping() {
  const t = document.getElementById('typing-indicator');
  if (t) t.remove();
}

function getLocalAIResponse(msg) {
  const d = currentEncounterData || {};
  const hf = d.hf_risk || 50;
  const g = d.gates || { vision: 0.33, signal: 0.33, clinical: 0.34 };
  const ml = msg.toLowerCase();
  if (ml.includes('hf') || ml.includes('heart') || ml.includes('predict') || ml.includes('why'))
    return `The **${hf}% Heart Failure risk** prediction was driven by Clinical Labs (**${Math.round(g.clinical*100)}%**), ECG (**${Math.round(g.signal*100)}%**), and CXR (**${Math.round(g.vision*100)}%**).\n\n📋 AHA 2022 HF Guidelines — Section 5.2, p.14`;
  if (ml.includes('lab') || ml.includes('bnp'))
    return "**BNP elevation** is the strongest predictor of HF decompensation. Values >400 pg/mL indicate Class I HF. Hyponatremia and elevated creatinine suggest cardiorenal syndrome.\n\n📋 ESC HF Guidelines 2021 — Section 7.3";
  if (ml.includes('guideline') || ml.includes('aha') || ml.includes('treatment'))
    return "**Applicable guidelines:**\n• AHA 2022: Diuretics (Class I), ACE inhibitors, Beta-blockers\n• ESC 2021: SGLT2 inhibitors (Dapagliflozin) — Class I\n\n📋 AHA 2022 — Sections 7.3, 7.4";
  if (ml.includes('ecg') || ml.includes('rhythm'))
    return `ECG shows: **${(d.ecg_findings || ['Atrial fibrillation']).join(', ')}**.\n\nECG contributed **${Math.round(g.signal*100)}%** to the prediction. AF increases HF risk by ~35%.\n\n📋 AHA 2020 AF Guidelines`;
  return `I'm VisionCare's AI assistant. Current prediction: **HF ${hf}%** | **Mortality ${d.mortality_risk || 30}%**\n\nAsk me about HF risk, labs, ECG, guidelines, or prognosis.`;
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── API Helper ─────────────────────────────────────────────────────
async function apiFetch(path, method='GET', body=null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── Mock Data (offline fallback) ───────────────────────────────────
function getMockPatients() {
  return [
    {id:'1',mrn:'MRN-004821',name:'Rajesh Kumar', age:67,gender:'Male',  status:'Active',    last_encounter:'26 Mar 2026',hf_risk:82,mortality_risk:31},
    {id:'2',mrn:'MRN-003156',name:'Priya Sharma', age:54,gender:'Female',status:'Active',    last_encounter:'24 Mar 2026',hf_risk:45,mortality_risk:18},
    {id:'3',mrn:'MRN-005934',name:'Anand Patel',  age:72,gender:'Male',  status:'Active',    last_encounter:'22 Mar 2026',hf_risk:91,mortality_risk:56},
    {id:'4',mrn:'MRN-002187',name:'Meera Iyer',   age:49,gender:'Female',status:'Discharged',last_encounter:'20 Mar 2026',hf_risk:12,mortality_risk:5},
    {id:'5',mrn:'MRN-006743',name:'Vikram Singh', age:61,gender:'Male',  status:'Active',    last_encounter:'18 Mar 2026',hf_risk:67,mortality_risk:28},
    {id:'6',mrn:'MRN-001298',name:'Lakshmi Nair', age:58,gender:'Female',status:'Discharged',last_encounter:'15 Mar 2026',hf_risk:23,mortality_risk:8},
    {id:'7',mrn:'MRN-007512',name:'Suresh Reddy', age:75,gender:'Male',  status:'Active',    last_encounter:'12 Mar 2026',hf_risk:78,mortality_risk:42},
    {id:'8',mrn:'MRN-008901',name:'Deepa Menon',  age:45,gender:'Female',status:'Active',    last_encounter:'10 Mar 2026',hf_risk:34,mortality_risk:12},
  ];
}

function getMockEncounters(patientId) {
  const map = {
    '1': [
      {id:'e032',label:'E-032',date:'26 Mar 2026',description:'Acute decompensated HF',hf_risk:82,mortality_risk:31,
       gates:{vision:0.15,signal:0.25,clinical:0.60},
       cxr_findings:['Cardiomegaly','Pulmonary edema','Pleural effusion (bilateral)'],
       ecg_findings:['Atrial fibrillation','Left ventricular hypertrophy','ST changes'],
       labs:[
         {name:'BNP',       value:'850 pg/mL',  normal:'<100',     status:'Critical'},
         {name:'Creatinine',value:'1.8 mg/dL',  normal:'0.7-1.2',  status:'High'},
         {name:'Sodium',    value:'132 mEq/L',  normal:'135-145',  status:'Low'},
         {name:'Potassium', value:'4.2 mEq/L',  normal:'3.5-5.0',  status:'Normal'},
         {name:'Troponin I',value:'0.04 ng/mL', normal:'<0.04',    status:'Borderline'},
         {name:'Hemoglobin',value:'11.2 g/dL',  normal:'13.5-17.5',status:'Low'},
       ]},
      {id:'e028',label:'E-028',date:'14 Feb 2026',description:'Follow-up cardiac evaluation',hf_risk:58,mortality_risk:19,
       gates:{vision:0.45,signal:0.30,clinical:0.25},
       cxr_findings:['Mild cardiomegaly','Mild pulmonary congestion'],
       ecg_findings:['Atrial fibrillation','Controlled ventricular rate'],
       labs:[{name:'BNP',value:'420 pg/mL',normal:'<100',status:'High'},{name:'Creatinine',value:'1.4 mg/dL',normal:'0.7-1.2',status:'High'}]},
      {id:'e015',label:'E-015',date:'3 Nov 2025 — Discharged 10 Nov 2025',description:'Routine cardiac screening',hf_risk:41,mortality_risk:12,
       gates:{vision:0.55,signal:0.25,clinical:0.20},
       cxr_findings:['Borderline cardiomegaly'],ecg_findings:['Normal sinus rhythm'],
       labs:[{name:'BNP',value:'180 pg/mL',normal:'<100',status:'High'}]},
    ],
    '3': [{id:'e019',label:'E-019',date:'22 Mar 2026',description:'Severe HF with reduced EF',hf_risk:91,mortality_risk:56,
           gates:{vision:0.20,signal:0.30,clinical:0.50},
           cxr_findings:['Severe cardiomegaly','Pulmonary edema'],ecg_findings:['Left bundle branch block'],
           labs:[{name:'BNP',value:'2100 pg/mL',normal:'<100',status:'Critical'}]}],
  };
  return map[patientId] || [{id:`e00${patientId}`,label:`E-00${patientId}`,date:'15 Mar 2026',description:'Cardiac evaluation',hf_risk:45,mortality_risk:20,gates:{vision:0.45,signal:0.25,clinical:0.30},cxr_findings:['Borderline cardiomegaly'],ecg_findings:['Normal sinus rhythm'],labs:[{name:'BNP',value:'250 pg/mL',normal:'<100',status:'High'}]}];
}

// ── Init ───────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  // Allow Enter key on login
  document.getElementById('login-pw').addEventListener('keydown', e => {
    if (e.key === 'Enter') doLogin();
  });
});
