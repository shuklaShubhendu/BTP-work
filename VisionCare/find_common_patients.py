"""
Find common patients across MIMIC-IV, MIMIC-CXR, and MIMIC-IV ECG
Works on Windows (no wget needed)
"""

import os
import requests
import gzip
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PHYSIONET_USER = "shubhendu9966"
PHYSIONET_PASS = "mS2E?amsRcC!G?a"

METADATA_DIR = Path("metadata")
METADATA_DIR.mkdir(exist_ok=True)

# Files to download
FILES = {
    "patients.csv.gz": "https://physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz",
    "admissions.csv.gz": "https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz",
    "cxr_metadata.csv.gz": "https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz",
    "cxr_chexpert.csv.gz": "https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz",
    "ecg_records.csv": "https://physionet.org/files/mimic-iv-ecg/1.0/record_list.csv",
}

# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_file(url, output_path):
    """Download file with authentication"""
    print(f"  Downloading {output_path.name}...", end=" ")
    
    if output_path.exists():
        print("✓ Already exists")
        return True
    
    try:
        response = requests.get(url, auth=(PHYSIONET_USER, PHYSIONET_PASS), stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Done ({output_path.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("🔍 Finding Common Patients Across MIMIC Datasets")
    print("=" * 60)
    
    # 1. Download metadata files
    print("\n📥 Downloading metadata files...")
    for filename, url in FILES.items():
        download_file(url, METADATA_DIR / filename)
    
    # 2. Load MIMIC-IV
    print("\n📊 Loading datasets...")
    patients = pd.read_csv(METADATA_DIR / "patients.csv.gz")
    admissions = pd.read_csv(METADATA_DIR / "admissions.csv.gz")
    mimic4_subjects = set(patients['subject_id'].unique())
    print(f"  MIMIC-IV: {len(mimic4_subjects):,} patients")
    
    # 3. Load MIMIC-CXR
    cxr_meta = pd.read_csv(METADATA_DIR / "cxr_metadata.csv.gz")
    cxr_subjects = set(cxr_meta['subject_id'].unique())
    print(f"  MIMIC-CXR: {len(cxr_subjects):,} patients")
    
    # 4. Load MIMIC-IV ECG
    ecg_records = pd.read_csv(METADATA_DIR / "ecg_records.csv")
    ecg_subjects = set(ecg_records['subject_id'].unique())
    print(f"  MIMIC-IV ECG: {len(ecg_subjects):,} patients")
    
    # 5. Find overlaps
    print("\n🔗 Finding overlaps...")
    
    common_iv_cxr = mimic4_subjects & cxr_subjects
    print(f"  MIMIC-IV ∩ CXR: {len(common_iv_cxr):,}")
    
    common_iv_ecg = mimic4_subjects & ecg_subjects
    print(f"  MIMIC-IV ∩ ECG: {len(common_iv_ecg):,}")
    
    common_cxr_ecg = cxr_subjects & ecg_subjects
    print(f"  CXR ∩ ECG: {len(common_cxr_ecg):,}")
    
    common_all = mimic4_subjects & cxr_subjects & ecg_subjects
    print(f"\n  ✅ ALL THREE: {len(common_all):,} patients")
    
    # 6. Save common subjects
    common_df = pd.DataFrame({'subject_id': sorted(list(common_all))})
    common_df.to_csv('common_subjects.csv', index=False)
    print(f"\n💾 Saved to common_subjects.csv")
    
    # 7. Mortality stats for common patients
    if len(common_all) > 0:
        admissions_common = admissions[admissions['subject_id'].isin(common_all)]
        mortality_per_patient = admissions_common.groupby('subject_id')['hospital_expire_flag'].max()
        mortality_rate = mortality_per_patient.mean()
        
        print(f"\n📈 Statistics for common patients:")
        print(f"  Total admissions: {len(admissions_common):,}")
        print(f"  Mortality rate: {mortality_rate*100:.1f}%")
        print(f"  Positive (died): {mortality_per_patient.sum():,}")
        print(f"  Negative (survived): {len(mortality_per_patient) - mortality_per_patient.sum():,}")
    
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
