"""Debug intersection issue."""
import pandas as pd
from pathlib import Path

data_dir = Path("D:/VisionCare/data")
df = pd.read_csv(data_dir / "interlinked_patients.csv")

# Get the sets
cvd = set(df[df['has_cvd_diagnosis']]['subject_id'])
xray = set(df[df['has_xray']]['subject_id'])
ecg = set(df[df['has_ecg']]['subject_id'])

print("=== PATIENT ID RANGES ===")
print(f"CVD patients: min={min(cvd):,}, max={max(cvd):,}")
print(f"X-ray patients: min={min(xray):,}, max={max(xray):,}")
print(f"ECG patients: min={min(ecg):,}, max={max(ecg):,}")

print("\n=== OVERLAPS ===")
print(f"CVD ∩ X-ray: {len(cvd & xray):,}")
print(f"CVD ∩ ECG: {len(cvd & ecg):,}")
print(f"X-ray ∩ ECG: {len(xray & ecg):,}")

# Check if ANY overlap exists
if len(xray & ecg) == 0:
    print("\n⚠️ NO OVERLAP between X-ray and ECG patients!")
    print("This means MIMIC-III Waveform and MIMIC-CXR have different patient populations.")
    print("\nPossible reasons:")
    print("1. MIMIC-III is from 2001-2012, MIMIC-CXR from 2011-2016")
    print("2. The waveform database only has ~10,000 matched patients")
    print("\nSolution: Use MIMIC-IV Waveform database instead!")
    print("Or focus on CVD + X-ray (2 modalities) without ECG")
else:
    print("\n✅ Some overlap exists!")
    
# What about CVD + X-ray only?
cvd_xray = cvd & xray
print(f"\n=== ALTERNATIVE: CVD + X-ray (2 modalities) ===")
print(f"Patients with CVD AND X-ray: {len(cvd_xray):,}")

# Those with cardiomegaly
cardio = set(df[df['has_cardiomegaly']]['subject_id'])
cvd_xray_cardio = cvd_xray & cardio
print(f"CVD + X-ray + Cardiomegaly: {len(cvd_xray_cardio):,}")
