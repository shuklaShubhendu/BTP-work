// src/components/TopBar.jsx
import { Search } from 'lucide-react';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function TopBar({ title, onLogout, showSearch = true }) {
  const [q, setQ] = useState('');
  const navigate = useNavigate();
  const handleSearch = (e) => {
    if (e.key === 'Enter' && q.trim()) navigate(`/patients?q=${encodeURIComponent(q)}`);
  };
  return (
    <header className="topbar">
      <div className="topbar-left">
        <h2 className="topbar-title">{title}</h2>
        {showSearch && (
          <div className="search-box">
            <Search size={14} color="var(--text-muted)" />
            <input placeholder="Search patients by MRN or name..." value={q}
              onChange={e => setQ(e.target.value)} onKeyDown={handleSearch} />
          </div>
        )}
      </div>
      <div className="topbar-right">
        <div className="user-info">
          <div className="user-name">Dr. Arjun Mehta</div>
          <div className="user-role">Cardiologist</div>
        </div>
        <div className="user-avatar">AM</div>
        <button className="btn-logout" onClick={onLogout}>Logout</button>
      </div>
    </header>
  );
}
