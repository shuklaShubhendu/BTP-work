// src/pages/Login.jsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Heart, Eye, EyeOff } from 'lucide-react';

export default function Login({ onLogin }) {
  const [email, setEmail]   = useState('');
  const [pw, setPw]         = useState('');
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async () => {
    if (!email.trim() || !pw.trim()) return;
    setLoading(true);
    await new Promise(r => setTimeout(r, 600));
    onLogin();
    navigate('/dashboard');
  };

  return (
    <div className="login-page">
      <div className="login-bg-grid" />
      <div className="login-scan" />
      <div className="login-card">
        <div className="login-logo-wrap">
          <Heart size={26} color="var(--green)" fill="none" strokeWidth={2} />
        </div>
        <div className="login-title">VisionCare 3.0</div>
        <div className="login-sub">8-Disease Prediction · Balanced Gates · Explainable AI</div>

        <div className="form-field">
          <label>Email</label>
          <input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="doctor@hospital.org" />
        </div>
        <div className="form-field">
          <label>Password</label>
          <div className="pw-field">
            <input type={showPw ? 'text' : 'password'} value={pw} onChange={e => setPw(e.target.value)}
              placeholder="Enter your password" onKeyDown={e => e.key === 'Enter' && handleLogin()} />
            <button className="pw-toggle" onClick={() => setShowPw(!showPw)}>
              {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
        </div>

        <button className="login-btn" onClick={handleLogin} disabled={loading || !email.trim() || !pw.trim()}>
          {loading ? 'Signing in...' : 'Sign In'}
        </button>
        <div className="login-footer">Clinical Decision Support System — IIIT Kottayam</div>
      </div>
    </div>
  );
}
