"""
VisionCare - Step 1: Find Interlinked Patients
===============================================
This script finds patients who exist in ALL THREE MIMIC datasets:
- MIMIC-IV (Clinical data)
- MIMIC-CXR (Chest X-rays)
- MIMIC-III Waveform (ECG signals)

Run this AFTER downloading the metadata files from PhysioNet.
"""

import os
import sys
import gzip
import pandas as pd
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration (paths point to D: drive)
try:
    from config import (
        DATA_ROOT, MIMIC_IV_DIR, MIMIC_CXR_DIR, MIMIC_WAVEFORM_DIR
    )
    DATA_DIR = Path(DATA_ROOT)
    MIMIC_IV_DIR = Path(MIMIC_IV_DIR)
    MIMIC_CXR_DIR = Path(MIMIC_CXR_DIR)
    MIMIC_WAVEFORM_DIR = Path(MIMIC_WAVEFORM_DIR)
except ImportError:
    print("⚠️  config.py not found, using default paths")
    DATA_DIR = Path("D:/VisionCare/data")
    MIMIC_IV_DIR = DATA_DIR / "raw" / "mimic_iv"
    MIMIC_CXR_DIR = DATA_DIR / "raw" / "mimic_cxr"
    MIMIC_WAVEFORM_DIR = DATA_DIR / "raw" / "mimic_waveform"

# Output
OUTPUT_FILE = DATA_DIR / "interlinked_patients.csv"


def find_file(directory: Path, pattern: str) -> Path:
    """Find a file matching partial name (handles .gz and non-.gz versions)."""
    directory = Path(directory)
    
    # Try exact patterns
    for suffix in ['', '.gz', '.csv', '.csv.gz']:
        candidate = directory / f"{pattern}{suffix}"
        if candidate.exists():
            return candidate
    
    # Try glob matching
    matches = list(directory.glob(f"*{pattern}*"))
    if matches:
        return matches[0]
    
    return None


def check_files_exist():
    """Verify all required metadata files are present."""
    print("\n🔍 Checking for required files...")
    
    required = {
        "MIMIC-IV diagnoses": find_file(MIMIC_IV_DIR, "diagnoses"),
        "MIMIC-IV patients": find_file(MIMIC_IV_DIR, "patients"),
        "MIMIC-CXR metadata": find_file(MIMIC_CXR_DIR, "metadata"),
        "MIMIC-CXR labels": find_file(MIMIC_CXR_DIR, "chexpert"),
        "Waveform RECORDS": find_file(MIMIC_WAVEFORM_DIR, "RECORDS"),
    }
    
    missing = []
    found = {}
    for name, path in required.items():
        if path and path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✅ {name}: {path.name} ({size_kb:.0f} KB)")
            found[name] = path
        else:
            missing.append(name)
            print(f"  ❌ {name}: NOT FOUND in {MIMIC_IV_DIR if 'IV' in name else MIMIC_CXR_DIR if 'CXR' in name else MIMIC_WAVEFORM_DIR}")
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} required files!")
        print("Please download them from PhysioNet.")
        return None
    
    print("\n✅ All required files found!")
    return found


def read_csv_file(path: Path) -> pd.DataFrame:
    """Read CSV file, handling both .gz and uncompressed."""
    path = Path(path)
    if path.suffix == '.gz':
        return pd.read_csv(path, compression='gzip')
    else:
        return pd.read_csv(path)


def get_cvd_patients_mimic_iv(diagnoses_path: Path):
    """Find patients with cardiovascular disease diagnoses."""
    print("\n📊 Loading MIMIC-IV diagnoses...")
    
    diagnoses = read_csv_file(diagnoses_path)
    print(f"   Total diagnosis records: {len(diagnoses):,}")
    
    # Filter for CVD ICD codes
    # ICD-10: I00-I99 (Circulatory system diseases)
    # ICD-9: 390-459 (Circulatory system)
    cvd_icd10 = diagnoses[diagnoses['icd_code'].str.match(r'^I[0-6]', na=False)]
    cvd_icd9 = diagnoses[diagnoses['icd_code'].str.match(r'^(39[0-9]|4[0-5][0-9])', na=False)]
    
    cvd_diagnoses = pd.concat([cvd_icd10, cvd_icd9]).drop_duplicates()
    cvd_patients = set(cvd_diagnoses['subject_id'].unique())
    
    print(f"   CVD diagnosis records: {len(cvd_diagnoses):,}")
    print(f"   ✅ CVD patients found: {len(cvd_patients):,}")
    
    return cvd_patients


def get_xray_patients(metadata_path: Path, labels_path: Path):
    """Find patients with chest X-rays."""
    print("\n🩻 Loading MIMIC-CXR data...")
    
    metadata = read_csv_file(metadata_path)
    labels = read_csv_file(labels_path)
    
    print(f"   Total X-ray studies: {len(metadata):,}")
    
    all_xray_patients = set(metadata['subject_id'].unique())
    
    # Merge to find Cardiomegaly cases
    merged = metadata.merge(labels, on=['subject_id', 'study_id'], how='left')
    cardiomegaly_patients = set(
        merged[merged.get('Cardiomegaly', 0) == 1.0]['subject_id'].unique()
    )
    
    print(f"   ✅ Patients with X-rays: {len(all_xray_patients):,}")
    print(f"   ✅ Patients with Cardiomegaly: {len(cardiomegaly_patients):,}")
    
    return all_xray_patients, cardiomegaly_patients


def get_ecg_patients(records_path: Path):
    """Find patients with ECG waveforms."""
    print("\n📈 Loading MIMIC-III Waveform records...")
    
    waveform_patients = set()
    record_count = 0
    
    with open(records_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record_count += 1
            parts = line.split('/')
            
            if len(parts) >= 2:
                subject_folder = parts[1]  # e.g., "p000033"
                try:
                    subject_id = int(subject_folder.lstrip('p'))
                    waveform_patients.add(subject_id)
                except ValueError:
                    continue
    
    print(f"   Total waveform records: {record_count:,}")
    print(f"   ✅ Patients with ECG: {len(waveform_patients):,}")
    
    return waveform_patients


def find_intersection(cvd_patients, xray_patients, ecg_patients, cardiomegaly_patients):
    """Find patients in all three datasets."""
    print("\n🔗 Computing intersections...")
    
    # Intersection of all three
    all_three = cvd_patients & xray_patients & ecg_patients
    xray_and_ecg = xray_patients & ecg_patients
    cardiomegaly_with_ecg = cardiomegaly_patients & ecg_patients
    
    print("\n" + "="*60)
    print("📊 INTERSECTION RESULTS")
    print("="*60)
    print(f"\n{'Dataset':<40} {'Count':>10}")
    print("-"*60)
    print(f"{'CVD patients (MIMIC-IV)':<40} {len(cvd_patients):>10,}")
    print(f"{'X-ray patients (MIMIC-CXR)':<40} {len(xray_patients):>10,}")
    print(f"{'ECG patients (Waveform)':<40} {len(ecg_patients):>10,}")
    print("-"*60)
    print(f"{'X-ray ∩ ECG (any diagnosis)':<40} {len(xray_and_ecg):>10,}")
    print(f"{'CVD ∩ X-ray ∩ ECG':<40} {len(all_three):>10,}")
    print(f"{'Cardiomegaly ∩ ECG':<40} {len(cardiomegaly_with_ecg):>10,}")
    print("="*60)
    
    return {
        'cvd_xray_ecg': all_three,
        'xray_ecg': xray_and_ecg,
        'cardiomegaly_ecg': cardiomegaly_with_ecg
    }


def save_results(intersections, cvd_patients, xray_patients, ecg_patients, cardiomegaly_patients):
    """Save results to CSV."""
    print(f"\n💾 Saving results...")
    
    # Build dataset of all patients
    all_patients = cvd_patients | xray_patients | ecg_patients
    
    data = []
    for pid in sorted(all_patients):
        data.append({
            'subject_id': pid,
            'has_cvd_diagnosis': pid in cvd_patients,
            'has_xray': pid in xray_patients,
            'has_ecg': pid in ecg_patients,
            'has_cardiomegaly': pid in cardiomegaly_patients,
            'has_all_three': pid in intersections['cvd_xray_ecg'],
        })
    
    df = pd.DataFrame(data)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Save interlinked patient IDs
    ids_file = DATA_DIR / "interlinked_patient_ids.txt"
    with open(ids_file, 'w') as f:
        for pid in sorted(intersections['cvd_xray_ecg']):
            f.write(f"{pid}\n")
    
    print(f"   ✅ Saved: {OUTPUT_FILE}")
    print(f"   ✅ Saved: {ids_file}")
    
    interlinked = df[df['has_all_three']]
    print(f"\n📋 Summary:")
    print(f"   Total interlinked patients: {len(interlinked)}")
    print(f"   With Cardiomegaly: {interlinked['has_cardiomegaly'].sum()}")
    
    return df


def main():
    print("="*60)
    print("🫀 VisionCare - Finding Interlinked Patients")
    print("="*60)
    
    # Find files
    found_files = check_files_exist()
    if not found_files:
        return None
    
    # Get patients from each dataset
    cvd_patients = get_cvd_patients_mimic_iv(found_files["MIMIC-IV diagnoses"])
    xray_patients, cardiomegaly = get_xray_patients(
        found_files["MIMIC-CXR metadata"],
        found_files["MIMIC-CXR labels"]
    )
    ecg_patients = get_ecg_patients(found_files["Waveform RECORDS"])
    
    # Find intersections
    intersections = find_intersection(cvd_patients, xray_patients, ecg_patients, cardiomegaly)
    
    # Save results
    df = save_results(intersections, cvd_patients, xray_patients, ecg_patients, cardiomegaly)
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print("Next: python scripts/02_download_patient_data.py")
    
    return df


if __name__ == "__main__":
    main()
