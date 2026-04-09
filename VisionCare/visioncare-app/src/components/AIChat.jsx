// src/components/AIChat.jsx
import { useState, useRef, useEffect } from 'react';
import { Bot, Send } from 'lucide-react';
import { askGemini, buildSystemPrompt, getRuleBasedResponse } from '../utils/gemini';

const SUGGESTIONS = [
  'Explain the Heart Failure risk',
  'What do the Lab findings indicate?',
  'Which AHA guideline applies?',
  'What is the mortality prognosis?',
];

function formatMsg(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')
    .replace(/📋 (.*)/g, '<span class="msg-citation">📋 $1</span>');
}

export default function AIChat({ encounter, patient }) {
  const [msgs, setMsgs] = useState([
    { type: 'system', text: 'AI is aware of this patient\'s prediction data' },
    { type: 'user',   text: 'Why did the AI predict this Heart Failure probability?' },
  ]);
  const [initialized, setInitialized] = useState(false);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  // Auto-load initial bot response
  useEffect(() => {
    if (!initialized && encounter) {
      setInitialized(true);
      const initial = getRuleBasedResponse('hf risk explain', encounter, patient);
      setMsgs(prev => [...prev, { type: 'bot', text: initial }]);
    }
  }, [encounter, initialized, patient]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [msgs]);

  const send = async (msg) => {
    if (!msg.trim() || loading) return;
    setInput('');
    setMsgs(prev => [...prev, { type: 'user', text: msg }]);
    setLoading(true);
    try {
      const system = buildSystemPrompt(encounter, patient);
      const geminiReply = await askGemini(system, msg);
      const reply = geminiReply || getRuleBasedResponse(msg, encounter, patient);
      setMsgs(prev => [...prev, { type: 'bot', text: reply }]);
    } catch {
      setMsgs(prev => [...prev, { type: 'bot', text: getRuleBasedResponse(msg, encounter, patient) }]);
    }
    setLoading(false);
  };

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-bot-icon"><Bot size={18} color="var(--green)" /></div>
        <div>
          <div className="chat-title">Medical AI Assistant</div>
          <div className="chat-subtitle">Powered by Gemini + Clinical Guidelines</div>
        </div>
      </div>

      <div className="chat-messages">
        {msgs.map((m, i) => (
          <div key={i} className={`chat-msg ${m.type}`}
            dangerouslySetInnerHTML={m.type === 'bot'
              ? { __html: formatMsg(m.text) }
              : undefined}>
            {m.type !== 'bot' ? m.text : undefined}
          </div>
        ))}
        {loading && (
          <div className="chat-msg bot">
            <div className="typing-indicator"><span/><span/><span/></div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-suggestions">
        {SUGGESTIONS.map(s => (
          <button key={s} className="chat-sug-btn" onClick={() => send(s)}>{s}</button>
        ))}
      </div>

      <div className="chat-input-row">
        <input
          value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send(input)}
          placeholder="Ask about this patient..."
        />
        <button className="chat-send-btn" onClick={() => send(input)}>
          <Send size={15} />
        </button>
      </div>
    </div>
  );
}
