// src/components/Sidebar.jsx
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Users, Microscope, BarChart3, Info, LogOut, Heart } from 'lucide-react';

const NAV = [
  { to: '/dashboard',  icon: LayoutDashboard, label: 'Dashboard'     },
  { to: '/patients',   icon: Users,            label: 'Patients'      },
  { to: '/analyze',    icon: Microscope,       label: 'Analysis Center'},
  { to: '/model',      icon: BarChart3,        label: 'Model Monitor' },
  { to: '/about',      icon: Info,             label: 'About'         },
];

export default function Sidebar({ onLogout }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-icon">
          <Heart size={18} color="var(--green)" />
        </div>
        <div>
          <div className="brand-name">VisionCare</div>
          <div className="brand-ver">v2.0</div>
        </div>
      </div>
      <nav className="sidebar-nav">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to} className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            <Icon size={17} className="nav-icon" />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div className="sf-inst">IIIT Kottayam</div>
        <div className="sf-prog">B.Tech CSE · Sem 7</div>
      </div>
    </aside>
  );
}
