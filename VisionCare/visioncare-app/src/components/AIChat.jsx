// src/components/AIChat.jsx — Floating Medical AI Chat (calls backend RAG + Gemini)
import { useState, useRef, useEffect } from 'react';
import { useMatch } from 'react-router-dom';
import { Bot, Send, X, MessageCircle, Sparkles } from 'lucide-react';
import { chat as chatAPI, getEncounter, getPatient } from '../utils/api';

const SUGGESTIONS = [
  'Explain all 8 disease risks',
  'Which disease risk is highest right now?',
  'Why is the Heart Failure risk high?',
  'What does the Sepsis risk mean?',
  'What do the lab values indicate?',
];

function formatMsg(text) {
  const safe = String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  return safe
    .replace(/^###\s*(.+)$/gm, '<span class="msg-section">$1</span>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/(\d{1,3}(?:\.\d+)?%)/g, '<span class="msg-percent">$1</span>')
    .replace(/\((Critical)\)/g, '<span class="risk-pill critical">$1</span>')
    .replace(/\((Moderate)\)/g, '<span class="risk-pill moderate">$1</span>')
    .replace(/\((Low)\)/g, '<span class="risk-pill low">$1</span>')
    .replace(/\n/g, '<br/>')
    .replace(/📋 (.*)/g, '<span class="msg-citation">📋 $1</span>');
}

export default function AIChat({ encounter, patient }) {
  const encounterMatch = useMatch('/patients/:patientId/encounters/:encounterId');
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState([
    { type: 'system', text: '🧠 VisionCare Medical AI — powered by RAG + Gemini 2.5 Flash' },
  ]);
  const [initialized, setInitialized] = useState(false);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [routePatient, setRoutePatient] = useState(null);
  const [routeEncounter, setRouteEncounter] = useState(null);
  const bottomRef = useRef(null);
  const activePatient = patient || routePatient;
  const activeEncounter = encounter || routeEncounter;

  useEffect(() => {
    let active = true;
    async function loadEncounterContext() {
      if (!encounterMatch) {
        if (active) {
          setRoutePatient(null);
          setRouteEncounter(null);
        }
        return;
      }

      try {
        const { patientId, encounterId } = encounterMatch.params;
        const [patientData, encounterData] = await Promise.all([
          getPatient(patientId),
          getEncounter(patientId, encounterId),
        ]);
        if (!active) return;
        setRoutePatient(patientData);
        setRouteEncounter(encounterData);
      } catch (err) {
        if (!active) return;
        setRoutePatient(null);
        setRouteEncounter(null);
      }
    }

    loadEncounterContext();
    return () => { active = false; };
  }, [encounterMatch?.params?.patientId, encounterMatch?.params?.encounterId]);

  useEffect(() => {
    setInitialized(false);
    setMsgs([
      { type: 'system', text: activeEncounter
        ? `🧠 VisionCare Medical AI — ready for ${activePatient?.name || 'this patient'} (${activeEncounter?.label || activeEncounter?.id || 'encounter'})`
        : '🧠 VisionCare Medical AI — powered by RAG + Gemini 2.5 Flash' },
    ]);
  }, [activePatient?.id, activeEncounter?.id]);

  // Auto-load initial bot response when opened with encounter context
  useEffect(() => {
    if (open && !initialized && activeEncounter) {
      setInitialized(true);
      setMsgs(prev => [...prev,
        { type: 'user', text: 'Give me an overview of this patient\'s risk assessment.' },
      ]);
      sendToBackend('Give me an overview of this patient\'s risk assessment.');
    }
  }, [open, activeEncounter, initialized]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [msgs]);

  const sendToBackend = async (msg) => {
    setLoading(true);
    try {
      const body = {
        message: msg,
        patient_id: activePatient?.id || null,
        encounter_id: activeEncounter?.id || null,
        // Inline context fallback
        patient_name: activePatient?.name,
        patient_age: activePatient?.age,
        patient_gender: activePatient?.gender,
        risks: activeEncounter?.risks,
        gates: activeEncounter?.gates,
        cxr_findings: activeEncounter?.cxr_findings,
        ecg_findings: activeEncounter?.ecg_findings,
        labs_data: activeEncounter?.labs,
      };
      const res = await chatAPI(body);
      const reply = res.reply || res.text || 'No response from AI.';
      const source = res.source || 'unknown';
      const chunks = res.chunks_retrieved || 0;
      const latency = res.latency_ms || 0;
      const fallbackReason = res.fallback_reason || '';

      setMsgs(prev => [...prev, {
        type: 'bot',
        text: reply,
        meta: { source, chunks, latency, fallbackReason },
      }]);
    } catch (err) {
      console.error('Chat error:', err);
      setMsgs(prev => [...prev, {
        type: 'bot',
        text: 'Unable to reach the Medical AI backend. Please ensure Docker is running.',
        meta: { source: 'error' },
      }]);
    }
    setLoading(false);
  };

  const send = async (msg) => {
    if (!msg.trim() || loading) return;
    setInput('');
    setMsgs(prev => [...prev, { type: 'user', text: msg }]);
    await sendToBackend(msg);
  };

  // Floating button + panel
  return (
    <>
      {/* Floating Toggle Button */}
      {!open && (
        <button className="chat-fab" onClick={() => setOpen(true)} id="chat-toggle-btn">
          <MessageCircle size={22} />
          <span className="chat-fab-pulse" />
        </button>
      )}

      {/* Chat Panel */}
      {open && (
        <div className="chat-floating-panel">
          {/* Header */}
          <div className="chat-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div className="chat-bot-icon"><Bot size={18} color="var(--green)" /></div>
              <div>
                <div className="chat-title">Medical AI Assistant</div>
                <div className="chat-subtitle">
                  <Sparkles size={10} style={{ marginRight: 3 }} />
                  RAG + Gemini 2.5 Flash · Medical domain only
                </div>
              </div>
            </div>
            <button className="chat-close-btn" onClick={() => setOpen(false)}>
              <X size={16} />
            </button>
          </div>

          {/* Messages */}
          <div className="chat-messages">
            {msgs.map((m, i) => (
              <div key={i} className={`chat-msg ${m.type}`}>
                {m.type === 'bot' ? (
                  <>
                    <div dangerouslySetInnerHTML={{ __html: formatMsg(m.text) }} />
                    {m.meta && (
                      <div className="chat-msg-meta">
                        {m.meta.source === 'gemini_rag' && <span className="chat-meta-badge rag">🧠 RAG + Gemini</span>}
                        {m.meta.source === 'rule_based' && <span className="chat-meta-badge rule">📋 Rule-based</span>}
                        {m.meta.source === 'error' && <span className="chat-meta-badge error">⚠️ Error</span>}
                        {m.meta.chunks > 0 && <span className="chat-meta-info">{m.meta.chunks} chunks</span>}
                        {m.meta.latency > 0 && <span className="chat-meta-info">{m.meta.latency}ms</span>}
                        {m.meta.fallbackReason && <span className="chat-meta-info">fallback: {m.meta.fallbackReason}</span>}
                      </div>
                    )}
                  </>
                ) : (
                  m.text
                )}
              </div>
            ))}
            {loading && (
              <div className="chat-msg bot">
                <div className="typing-indicator"><span/><span/><span/></div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                  Retrieving from medical textbooks...
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Suggestions */}
          <div className="chat-suggestions">
            {SUGGESTIONS.map(s => (
              <button key={s} className="chat-sug-btn" onClick={() => send(s)}>{s}</button>
            ))}
          </div>

          {/* Input */}
          <div className="chat-input-row">
            <input
              value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && send(input)}
              placeholder={activeEncounter ? "Ask about this patient's risks..." : 'Ask about model behavior, labs, or clinical findings...'}
            />
            <button className="chat-send-btn" onClick={() => send(input)} disabled={loading}>
              <Send size={15} />
            </button>
          </div>
        </div>
      )}
    </>
  );
}
