// src/App.jsx
import { useState } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import Sidebar       from './components/Sidebar';
import TopBar        from './components/TopBar';
import Login         from './pages/Login';
import Dashboard     from './pages/Dashboard';
import Patients      from './pages/Patients';
import PatientProfile   from './pages/PatientProfile';
import EncounterAnalysis from './pages/EncounterAnalysis';
import AnalysisCenter   from './pages/AnalysisCenter';
import ModelMonitor     from './pages/ModelMonitor';
import About            from './pages/About';

const PAGE_TITLES = {
  '/dashboard': 'Dashboard',
  '/patients':  'Patients',
  '/analyze':   'Analysis Center',
  '/model':     'Model Monitor',
  '/about':     'About',
};

function AppShell({ children, onLogout }) {
  const loc = useLocation();
  const isEncounter = loc.pathname.includes('/encounters/');
  const isPatient = !isEncounter && loc.pathname.includes('/patients/') && loc.pathname.split('/').length > 3;
  const title = PAGE_TITLES[loc.pathname] || (isEncounter ? 'Encounter Analysis' : isPatient ? 'Patient Profile' : 'VisionCare');
  return (
    <div className="app-layout">
      <Sidebar onLogout={onLogout} />
      <div className="main-area">
        <TopBar title={title} onLogout={onLogout} showSearch={!isEncounter} />
        {children}
      </div>
    </div>
  );
}

function PrivateRoute({ authed, children }) {
  return authed ? children : <Navigate to="/login" replace />;
}

export default function App() {
  const [authed, setAuthed] = useState(false);
  return (
    <Routes>
      <Route path="/login" element={<Login onLogin={() => setAuthed(true)} />} />
      <Route path="/*" element={
        <PrivateRoute authed={authed}>
          <AppShell onLogout={() => setAuthed(false)}>
            <Routes>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/patients"  element={<Patients />} />
              <Route path="/patients/:patientId" element={<PatientProfile />} />
              <Route path="/patients/:patientId/encounters/:encounterId" element={<EncounterAnalysis />} />
              <Route path="/analyze"   element={<AnalysisCenter />} />
              <Route path="/model"     element={<ModelMonitor />} />
              <Route path="/about"     element={<About />} />
              <Route path="*"          element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </AppShell>
        </PrivateRoute>
      } />
      <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
  );
}
