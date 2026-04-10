// src/components/AIChat.jsx — Floating Medical AI Chat (calls backend RAG + Gemini)
import { useState, useRef, useEffect } from 'react';
import { Bot, Send, X, MessageCircle, Sparkles } from 'lucide-react';
import { chat as chatAPI } from '../utils/api';

const SUGGESTIONS = [
  'Explain all 8 disease risks',
  'Why is the Heart Failure risk high?',
  'What does the Sepsis risk mean?',
  'Explain the gate weight balance',
  'What do the lab values indicate?',
];

function formatMsg(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')
    .replace(/📋 (.*)/g, '<span class="msg-citation">📋 $1</span>');
}

export default function AIChat({ encounter, patient }) {
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState([
    { type: 'system', text: '🧠 VisionCare Medical AI — powered by RAG + Gemini 2.5 Flash' },
  ]);
  const [initialized, setInitialized] = useState(false);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  // Auto-load initial bot response when opened with encounter context
  useEffect(() => {
    if (open && !initialized && encounter) {
      setInitialized(true);
      setMsgs(prev => [...prev,
        { type: 'user', text: 'Give me an overview of this patient\'s risk assessment.' },
      ]);
      sendToBackend('Give me an overview of this patient\'s risk assessment.');
    }
  }, [open, encounter, initialized]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [msgs]);

  const sendToBackend = async (msg) => {
    setLoading(true);
    try {
      const body = {
        message: msg,
        patient_id: patient?.id || null,
        encounter_id: encounter?.id || null,
        // Inline context fallback
        patient_name: patient?.name,
        patient_age: patient?.age,
        patient_gender: patient?.gender,
        risks: encounter?.risks,
        gates: encounter?.gates,
        cxr_findings: encounter?.cxr_findings,
        ecg_findings: encounter?.ecg_findings,
        labs_data: encounter?.labs,
      };
      const res = await chatAPI(body);
      const reply = res.reply || res.text || 'No response from AI.';
      const source = res.source || 'unknown';
      const chunks = res.chunks_retrieved || 0;
      const latency = res.latency_ms || 0;

      setMsgs(prev => [...prev, {
        type: 'bot',
        text: reply,
        meta: { source, chunks, latency },
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
              placeholder="Ask about this patient's risks..."
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
