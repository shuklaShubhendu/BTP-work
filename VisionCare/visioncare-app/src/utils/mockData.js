// src/utils/mockData.js — V3 Full mock dataset (8 diseases + balanced gates)
const IMG = (name) => `http://127.0.0.1:8000/images/${name}`;

// V3 target disease names
export const V3_TARGETS = [
  { key: 'mortality',              label: 'Mortality',              short: 'MORT' },
  { key: 'heart_failure',          label: 'Heart Failure',          short: 'HF' },
  { key: 'myocardial_infarction',  label: 'Myocardial Infarction', short: 'MI' },
  { key: 'arrhythmia',             label: 'Arrhythmia',             short: 'ARR' },
  { key: 'sepsis',                 label: 'Sepsis',                 short: 'SEP' },
  { key: 'pulmonary_embolism',     label: 'Pulmonary Embolism',     short: 'PE' },
  { key: 'acute_kidney_injury',    label: 'Acute Kidney Injury',    short: 'AKI' },
  { key: 'icu_admission',          label: 'ICU Admission',          short: 'ICU' },
];

export const MOCK_PATIENTS = [
  { id:'1', mrn:'MRN-004821', name:'Rajesh Kumar',  age:67, gender:'Male',   status:'Active',     condition:'Acute decompensated HF',      severity:'critical' },
  { id:'2', mrn:'MRN-003156', name:'Priya Sharma',  age:54, gender:'Female', status:'Active',     condition:'Follow-up cardiac evaluation', severity:'moderate' },
  { id:'3', mrn:'MRN-005934', name:'Anand Patel',   age:72, gender:'Male',   status:'Active',     condition:'Severe HF with reduced EF',    severity:'critical' },
  { id:'4', mrn:'MRN-002187', name:'Meera Iyer',    age:49, gender:'Female', status:'Discharged', condition:'Routine cardiac screening',    severity:'normal'   },
  { id:'5', mrn:'MRN-006743', name:'Vikram Singh',  age:61, gender:'Male',   status:'Active',     condition:'Moderate HF follow-up',        severity:'moderate' },
  { id:'6', mrn:'MRN-001298', name:'Lakshmi Nair',  age:58, gender:'Female', status:'Discharged', condition:'Post-MI cardiac evaluation',   severity:'normal'   },
  { id:'7', mrn:'MRN-007512', name:'Suresh Reddy',  age:75, gender:'Male',   status:'Active',     condition:'Chronic HF management',        severity:'critical' },
  { id:'8', mrn:'MRN-008901', name:'Deepa Menon',   age:45, gender:'Female', status:'Active',     condition:'New onset dyspnoea workup',    severity:'moderate' },
];

export const MOCK_ENCOUNTERS = {
  '1': [
    { id:'e032', label:'E-032', date:'26 Mar 2026', description:'Acute decompensated HF',
      risks: { mortality:31, heart_failure:82, myocardial_infarction:15, arrhythmia:48, sepsis:22, pulmonary_embolism:7, acute_kidney_injury:41, icu_admission:55 },
      gates: { vision:0.339, signal:0.338, clinical:0.323 },
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
    { id:'e028', label:'E-028', date:'14 Feb 2026', description:'Follow-up cardiac evaluation',
      risks: { mortality:19, heart_failure:58, myocardial_infarction:8, arrhythmia:32, sepsis:10, pulmonary_embolism:4, acute_kidney_injury:25, icu_admission:30 },
      gates: { vision:0.350, signal:0.330, clinical:0.320 },
      cxr_findings:['Mild cardiomegaly','Mild pulmonary congestion'],
      ecg_findings:['Atrial fibrillation','Controlled ventricular rate'],
      labs:[
        {name:'BNP',       value:'420 pg/mL', normal:'<100',    status:'High'},
        {name:'Creatinine',value:'1.4 mg/dL', normal:'0.7-1.3', status:'High'},
        {name:'Sodium',    value:'140 mEq/L', normal:'136-145', status:'Normal'},
      ],
      cxr_image: IMG('p1_e028.png'),
    },
  ],
  '2': [{
    id:'e041', label:'E-041', date:'24 Mar 2026', description:'Follow-up cardiac evaluation',
    risks: { mortality:18, heart_failure:45, myocardial_infarction:6, arrhythmia:22, sepsis:8, pulmonary_embolism:3, acute_kidney_injury:15, icu_admission:20 },
    gates: { vision:0.345, signal:0.330, clinical:0.325 },
    cxr_findings:['Mild cardiomegaly'], ecg_findings:['Normal sinus rhythm','Left axis deviation'],
    labs:[{name:'BNP',value:'290 pg/mL',normal:'<100',status:'High'},{name:'Creatinine',value:'0.9 mg/dL',normal:'0.7-1.3',status:'Normal'}],
    cxr_image: IMG('p2_e041.png'),
  }],
  '3': [{
    id:'e019', label:'E-019', date:'22 Mar 2026', description:'Severe HF with reduced EF',
    risks: { mortality:56, heart_failure:91, myocardial_infarction:28, arrhythmia:62, sepsis:35, pulmonary_embolism:12, acute_kidney_injury:64, icu_admission:72 },
    gates: { vision:0.320, signal:0.345, clinical:0.335 },
    cxr_findings:['Severe cardiomegaly','Pulmonary oedema','Pleural effusion'], ecg_findings:['Left bundle branch block','ST depression'],
    labs:[{name:'BNP',value:'2100 pg/mL',normal:'<100',status:'Critical'},{name:'Creatinine',value:'2.4 mg/dL',normal:'0.7-1.3',status:'Critical'},{name:'Sodium',value:'128 mEq/L',normal:'136-145',status:'Critical'}],
    cxr_image: IMG('p3_e019.png'),
  }],
  '4': [{
    id:'e004', label:'E-004', date:'20 Mar 2026', description:'Routine cardiac screening',
    risks: { mortality:5, heart_failure:12, myocardial_infarction:3, arrhythmia:8, sepsis:4, pulmonary_embolism:2, acute_kidney_injury:6, icu_admission:7 },
    gates: { vision:0.340, signal:0.335, clinical:0.325 },
    cxr_findings:['No significant findings'], ecg_findings:['Normal sinus rhythm'],
    labs:[{name:'BNP',value:'45 pg/mL',normal:'<100',status:'Normal'}],
    cxr_image: IMG('p4_e004.png'),
  }],
  '5': [{
    id:'e005', label:'E-005', date:'18 Mar 2026', description:'Moderate HF follow-up',
    risks: { mortality:28, heart_failure:67, myocardial_infarction:12, arrhythmia:38, sepsis:18, pulmonary_embolism:6, acute_kidney_injury:32, icu_admission:40 },
    gates: { vision:0.335, signal:0.340, clinical:0.325 },
    cxr_findings:['Cardiomegaly','Mild pulmonary congestion'], ecg_findings:['Atrial fibrillation','Rate-controlled'],
    labs:[{name:'BNP',value:'580 pg/mL',normal:'<100',status:'Critical'}],
    cxr_image: IMG('p5_e005.png'),
  }],
  '6': [{
    id:'e006', label:'E-006', date:'15 Mar 2026', description:'Post-MI cardiac evaluation',
    risks: { mortality:8, heart_failure:23, myocardial_infarction:45, arrhythmia:28, sepsis:6, pulmonary_embolism:5, acute_kidney_injury:12, icu_admission:18 },
    gates: { vision:0.330, signal:0.350, clinical:0.320 },
    cxr_findings:['Mild atelectasis'], ecg_findings:['Q waves in V1-V3','ST elevation'],
    labs:[{name:'BNP',value:'120 pg/mL',normal:'<100',status:'High'},{name:'Troponin I',value:'2.8 ng/mL',normal:'<0.04',status:'Critical'}],
    cxr_image: IMG('p6_e006.png'),
  }],
  '7': [{
    id:'e007', label:'E-007', date:'12 Mar 2026', description:'Chronic HF management',
    risks: { mortality:42, heart_failure:78, myocardial_infarction:18, arrhythmia:55, sepsis:25, pulmonary_embolism:8, acute_kidney_injury:52, icu_admission:58 },
    gates: { vision:0.325, signal:0.340, clinical:0.335 },
    cxr_findings:['Cardiomegaly','Pleural effusion'], ecg_findings:['Ventricular tachycardia','LV hypertrophy'],
    labs:[{name:'BNP',value:'1200 pg/mL',normal:'<100',status:'Critical'}],
    cxr_image: IMG('p7_e007.png'),
  }],
  '8': [{
    id:'e008', label:'E-008', date:'10 Mar 2026', description:'New onset dyspnea workup',
    risks: { mortality:12, heart_failure:34, myocardial_infarction:7, arrhythmia:18, sepsis:15, pulmonary_embolism:22, acute_kidney_injury:10, icu_admission:20 },
    gates: { vision:0.330, signal:0.325, clinical:0.345 },
    cxr_findings:['Borderline heart size'], ecg_findings:['Sinus tachycardia'],
    labs:[{name:'BNP',value:'220 pg/mL',normal:'<100',status:'High'},{name:'D-Dimer',value:'1.8 ug/mL',normal:'<0.5',status:'Critical'}],
    cxr_image: IMG('p8_e008.png'),
  }],
};

export const getPatientWithDetails = (id) => {
  const p = MOCK_PATIENTS.find(p => p.id === id);
  const encs = MOCK_ENCOUNTERS[id] || [];
  const latest = encs[0] || {};
  return {
    ...p,
    last_encounter: latest.date,
    risks: latest.risks || {},
    gates: latest.gates || {},
    hf_risk: latest.risks?.heart_failure || 0,
    mortality_risk: latest.risks?.mortality || 0,
  };
};
