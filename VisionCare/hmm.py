import os
import pandas as pd

PHYSIONET_USER =  "shubhendu9966" # CHANGE
PHYSIONET_PASS = "mS2E?amsRcC!G?a"  # CHANGE

os.makedirs('metadata', exist_ok=True)

# ============================================================
# 1. DOWNLOAD METADATA (Correct URLs)
# ============================================================

# MIMIC-IV patients
!wget -q --user {PHYSIONET_USER} --password {PHYSIONET_PASS} \
    -O metadata/patients.csv.gz \
    "https://physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz"

# MIMIC-IV admissions (has mortality!)
!wget -q --user {PHYSIONET_USER} --password {PHYSIONET_PASS} \
    -O metadata/admissions.csv.gz \
    "https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"

# MIMIC-CXR metadata (SEPARATE DATASET!)
!wget -q --user {PHYSIONET_USER} --password {PHYSIONET_PASS} \
    -O metadata/cxr_metadata.csv.gz \
    "https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz"

# MIMIC-CXR labels (chexpert)
!wget -q --user {PHYSIONET_USER} --password {PHYSIONET_PASS} \
    -O metadata/cxr_chexpert.csv.gz \
    "https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz"

# MIMIC-IV ECG records
!wget -q --user {PHYSIONET_USER} --password {PHYSIONET_PASS} \
    -O metadata/ecg_records.csv \
    "https://physionet.org/files/mimic-iv-ecg/1.0/record_list.csv"

!ls -lh metadata/
print("✓ Downloaded metadata files")

# ============================================================
# 2. LOAD AND FIND COMMON PATIENTS
# ============================================================

# MIMIC-IV
patients = pd.read_csv('metadata/patients.csv.gz')
admissions = pd.read_csv('metadata/admissions.csv.gz')
mimic4_subjects = set(patients['subject_id'].unique())
print(f"MIMIC-IV: {len(mimic4_subjects):,} patients")

# MIMIC-CXR
cxr_meta = pd.read_csv('metadata/cxr_metadata.csv.gz')
cxr_subjects = set(cxr_meta['subject_id'].unique())
print(f"MIMIC-CXR: {len(cxr_subjects):,} patients")

# MIMIC-IV ECG
ecg_records = pd.read_csv('metadata/ecg_records.csv')
ecg_subjects = set(ecg_records['subject_id'].unique())
print(f"MIMIC-IV ECG: {len(ecg_subjects):,} patients")

# ============================================================
# 3. FIND OVERLAP
# ============================================================

common_all = mimic4_subjects & cxr_subjects & ecg_subjects
print(f"\n✅ COMMON (all 3): {len(common_all):,} patients")

# Save
pd.DataFrame({'subject_id': list(common_all)}).to_csv('common_subjects.csv', index=False)

# Mortality stats
admissions_common = admissions[admissions['subject_id'].isin(common_all)]
mortality_rate = admissions_common.groupby('subject_id')['hospital_expire_flag'].max().mean()
print(f"📊 Mortality rate: {mortality_rate*100:.1f}%")