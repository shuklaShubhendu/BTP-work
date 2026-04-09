// src/utils/api.js
import axios from 'axios';

const API = axios.create({ baseURL: 'http://127.0.0.1:8000' });

export const getStats      = ()      => API.get('/api/stats').then(r => r.data);
export const getPatients   = (q='')  => API.get(`/api/patients?search=${q}`).then(r => r.data);
export const getPatient    = (id)    => API.get(`/api/patients/${id}`).then(r => r.data);
export const getEncounters = (id)    => API.get(`/api/patients/${id}/encounters`).then(r => r.data);
export const getEncounter  = (p,e)   => API.get(`/api/patients/${p}/encounters/${e}`).then(r => r.data);
export const chat          = (body)  => API.post('/api/chat', body).then(r => r.data);
export const getModelInfo  = ()      => API.get('/api/model/info').then(r => r.data);
export const getModelsStatus = ()    => API.get('/api/models/status').then(r => r.data);

// Multi-modal fusion
export const analyze       = (form)  => API.post('/api/analyze', form).then(r => r.data);

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
