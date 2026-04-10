// src/utils/api.js — V3 API client
import axios from 'axios';

const API = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
});

export const getStats      = ()      => API.get('/api/stats').then(r => r.data);
export const getPatients   = (q='')  => API.get(`/api/patients?search=${q}`).then(r => r.data);
export const getPatient    = (id)    => API.get(`/api/patients/${id}`).then(r => r.data);
export const getEncounters = (id)    => API.get(`/api/patients/${id}/encounters`).then(r => r.data);
export const getEncounter  = (p,e)   => API.get(`/api/patients/${p}/encounters/${e}`).then(r => r.data);
export const chat          = (body)  => API.post('/api/chat', body).then(r => r.data);
export const ragStatus     = ()      => API.get('/api/rag/status').then(r => r.data);
export const getModelInfo  = ()      => API.get('/api/model/info').then(r => r.data);
export const getModelsStatus = ()    => API.get('/api/models/status').then(r => r.data);

// V3 Multi-modal fusion (returns 8 diseases + 3 gate weights)
export const analyze       = (form)  => API.post('/api/analyze', form).then(r => r.data);
export const createPatient = (body)  => API.post('/api/patients', body).then(r => r.data);
export const onboardPatient = ({ patient, labs, hasEcg, cxrFile }) => {
  const fd = new FormData();
  fd.append('name', patient.name);
  fd.append('age', String(patient.age));
  fd.append('gender', patient.gender);
  fd.append('condition', patient.condition || '');
  fd.append('status', patient.status || 'Active');
  fd.append('has_ecg', String(Boolean(hasEcg)));
  fd.append('labs_json', JSON.stringify(labs || {}));
  if (cxrFile) {
    fd.append('cxr_file', cxrFile);
  }
  return API.post('/api/onboard', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data);
};

// V3 Predict specific encounter
export const predictEncounter = (encId) => API.post(`/api/predict/${encId}`).then(r => r.data);

// Single-modal endpoints
export const analyzeCxr = (file) => {
  const fd = new FormData();
  fd.append('file', file);
  return API.post('/api/analyze/cxr', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data);
};

export const analyzeEcg  = ()     => API.post('/api/analyze/ecg').then(r => r.data);
export const analyzeLabs = (labs) => API.post('/api/analyze/labs', { labs }).then(r => r.data);
