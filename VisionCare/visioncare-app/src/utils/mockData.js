// src/utils/mockData.js — Full mock dataset
// CXR images served from FastAPI backend at :8000/images/
// File naming: p{patient_id}_e{encounter_id}.png (NIH ChestX-ray14 source)
const IMG = (name) => `http://127.0.0.1:8000/images/${name}`;

export const MOCK_PATIENTS = [
  { id:'1', mrn:'MRN-004821', name:'Rajesh Kumar',  age:67, gender:'Male',   status:'Active',     condition:'Acute decompensated HF',      hf_risk:82, mortality_risk:31, severity:'critical' },
  { id:'2', mrn:'MRN-003156', name:'Priya Sharma',  age:54, gender:'Female', status:'Active',     condition:'Follow-up cardiac evaluation', hf_risk:45, mortality_risk:18, severity:'moderate' },
  { id:'3', mrn:'MRN-005934', name:'Anand Patel',   age:72, gender:'Male',   status:'Active',     condition:'Severe HF with reduced EF',    hf_risk:91, mortality_risk:56, severity:'critical' },
  { id:'4', mrn:'MRN-002187', name:'Meera Iyer',    age:49, gender:'Female', status:'Discharged', condition:'Routine cardiac screening',    hf_risk:12, mortality_risk:5,  severity:'normal'   },
  { id:'5', mrn:'MRN-006743', name:'Vikram Singh',  age:61, gender:'Male',   status:'Active',     condition:'Moderate HF follow-up',        hf_risk:67, mortality_risk:28, severity:'moderate' },
  { id:'6', mrn:'MRN-001298', name:'Lakshmi Nair',  age:58, gender:'Female', status:'Discharged', condition:'Post-MI cardiac evaluation',   hf_risk:23, mortality_risk:8,  severity:'normal'   },
  { id:'7', mrn:'MRN-007512', name:'Suresh Reddy',  age:75, gender:'Male',   status:'Active',     condition:'Chronic HF management',        hf_risk:78, mortality_risk:42, severity:'critical' },
  { id:'8', mrn:'MRN-008901', name:'Deepa Menon',   age:45, gender:'Female', status:'Active',     condition:'New onset dyspnoea workup',    hf_risk:34, mortality_risk:12, severity:'moderate' },
];

export const MOCK_ENCOUNTERS = {
  '1': [
    { id:'e032', label:'E-032', date:'26 Mar 2026', description:'Acute decompensated HF', hf_risk:82, mortality_risk:31,
      gates:{ vision:0.15, signal:0.25, clinical:0.60 },
      cxr_findings:['Cardiomegaly','Pulmonary oedema','Bilateral pleural effusion'],
      ecg_findings:['Atrial fibrillation','Left ventricular hypertrophy','ST changes'],
      labs:[
        {name:'BNP',       value:'850 pg/mL',   normal:'<100',     status:'Critical'},
        {name:'Troponin I',value:'9.45 ng/mL',  normal:'<0.04',    status:'Critical'},
        {name:'Creatinine',value:'1.8 mg/dL',   normal:'0.7-1.3',  status:'High'},
        {name:'Sodium',    value:'138 mEq/L',   normal:'136-145',  status:'Normal'},
        {name:'Potassium', value:'4.2 mEq/L',   normal:'3.5-5.0',  status:'Normal'},
        {name:'Hemoglobin',value:'11.2 g/dL',   normal:'13.5-17.5',status:'Low'},
        {name:'WBC',       value:'9.8 ×10³/μL', normal:'4.5-11.0', status:'Normal'},
        {name:'Glucose',   value:'142 mg/dL',   normal:'70-100',   status:'High'},
      ],
      cxr_image: IMG('p1_e032.png'),
    },
    { id:'e028', label:'E-028', date:'14 Feb 2026', description:'Follow-up cardiac evaluation', hf_risk:58, mortality_risk:19,
      gates:{ vision:0.45, signal:0.30, clinical:0.25 },
      cxr_findings:['Mild cardiomegaly','Mild pulmonary congestion'],
      ecg_findings:['Atrial fibrillation','Controlled ventricular rate'],
      labs:[
        {name:'BNP',       value:'420 pg/mL', normal:'<100',    status:'High'},
        {name:'Creatinine',value:'1.4 mg/dL', normal:'0.7-1.3', status:'High'},
        {name:'Sodium',    value:'140 mEq/L', normal:'136-145', status:'Normal'},
      ],
      cxr_image: IMG('p1_e028.png'),
    },
    { id:'e015', label:'E-015', date:'3 Nov 2025', description:'Routine cardiac screening', hf_risk:41, mortality_risk:12,
      gates:{ vision:0.55, signal:0.25, clinical:0.20 },
      cxr_findings:['Borderline cardiomegaly'],
      ecg_findings:['Normal sinus rhythm'],
      labs:[{name:'BNP',value:'180 pg/mL',normal:'<100',status:'High'}],
      cxr_image: IMG('p1_e028.png'), // reuse
    },
  ],
  '2': [{
    id:'e041', label:'E-041', date:'24 Mar 2026', description:'Follow-up cardiac evaluation', hf_risk:45, mortality_risk:18,
    gates:{vision:0.50, signal:0.20, clinical:0.30},
    cxr_findings:['Mild cardiomegaly'], ecg_findings:['Normal sinus rhythm','Left axis deviation'],
    labs:[{name:'BNP',value:'290 pg/mL',normal:'<100',status:'High'},{name:'Creatinine',value:'0.9 mg/dL',normal:'0.7-1.3',status:'Normal'}],
    cxr_image: IMG('p2_e041.png'),
  }],
  '3': [{
    id:'e019', label:'E-019', date:'22 Mar 2026', description:'Severe HF with reduced EF', hf_risk:91, mortality_risk:56,
    gates:{vision:0.20, signal:0.30, clinical:0.50},
    cxr_findings:['Severe cardiomegaly','Pulmonary oedema','Pleural effusion'], ecg_findings:['Left bundle branch block','ST depression'],
    labs:[{name:'BNP',value:'2100 pg/mL',normal:'<100',status:'Critical'},{name:'Creatinine',value:'2.4 mg/dL',normal:'0.7-1.3',status:'Critical'},{name:'Sodium',value:'128 mEq/L',normal:'136-145',status:'Critical'}],
    cxr_image: IMG('p3_e019.png'),
  }],
  '4': [{
    id:'e004', label:'E-004', date:'20 Mar 2026', description:'Routine cardiac screening', hf_risk:12, mortality_risk:5,
    gates:{vision:0.65, signal:0.20, clinical:0.15},
    cxr_findings:['No significant findings'], ecg_findings:['Normal sinus rhythm'],
    labs:[{name:'BNP',value:'45 pg/mL',normal:'<100',status:'Normal'},{name:'Creatinine',value:'0.8 mg/dL',normal:'0.7-1.3',status:'Normal'}],
    cxr_image: IMG('p4_e004.png'),
  }],
  '5': [{
    id:'e005', label:'E-005', date:'18 Mar 2026', description:'Moderate HF follow-up', hf_risk:67, mortality_risk:28,
    gates:{vision:0.35, signal:0.30, clinical:0.35},
    cxr_findings:['Cardiomegaly','Mild pulmonary congestion'], ecg_findings:['Atrial fibrillation','Rate-controlled'],
    labs:[{name:'BNP',value:'580 pg/mL',normal:'<100',status:'Critical'},{name:'Creatinine',value:'1.6 mg/dL',normal:'0.7-1.3',status:'High'}],
    cxr_image: IMG('p5_e005.png'),
  }],
  '6': [{
    id:'e006', label:'E-006', date:'15 Mar 2026', description:'Post-MI cardiac evaluation', hf_risk:23, mortality_risk:8,
    gates:{vision:0.55, signal:0.25, clinical:0.20},
    cxr_findings:['Mild atelectasis'], ecg_findings:['Q waves in V1-V3','Normal sinus rhythm'],
    labs:[{name:'BNP',value:'120 pg/mL',normal:'<100',status:'High'},{name:'Troponin I',value:'0.02 ng/mL',normal:'<0.04',status:'Normal'}],
    cxr_image: IMG('p6_e006.png'),
  }],
  '7': [{
    id:'e007', label:'E-007', date:'12 Mar 2026', description:'Chronic HF management', hf_risk:78, mortality_risk:42,
    gates:{vision:0.25, signal:0.35, clinical:0.40},
    cxr_findings:['Cardiomegaly','Pleural effusion'], ecg_findings:['Ventricular tachycardia','LV hypertrophy'],
    labs:[{name:'BNP',value:'1200 pg/mL',normal:'<100',status:'Critical'},{name:'Creatinine',value:'2.1 mg/dL',normal:'0.7-1.3',status:'Critical'}],
    cxr_image: IMG('p7_e007.png'),
  }],
  '8': [{
    id:'e008', label:'E-008', date:'10 Mar 2026', description:'New onset dyspnea workup', hf_risk:34, mortality_risk:12,
    gates:{vision:0.45, signal:0.25, clinical:0.30},
    cxr_findings:['Borderline heart size','Mild increased vascularity'], ecg_findings:['Normal sinus rhythm','Mild ST changes'],
    labs:[{name:'BNP',value:'220 pg/mL',normal:'<100',status:'High'},{name:'Creatinine',value:'1.0 mg/dL',normal:'0.7-1.3',status:'Normal'}],
    cxr_image: IMG('p8_e008.png'),
  }],
};

export const getPatientWithDetails = (id) => {
  const p = MOCK_PATIENTS.find(p => p.id === id);
  const encs = MOCK_ENCOUNTERS[id] || [];
  const latest = encs[0] || {};
  return { ...p, last_encounter: latest.date, hf_risk: latest.hf_risk, mortality_risk: latest.mortality_risk };
};
