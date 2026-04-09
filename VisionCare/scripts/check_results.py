"""Check interlinked patient results."""
import pandas as pd
from pathlib import Path

data_dir = Path("D:/VisionCare/data")
df = pd.read_csv(data_dir / "interlinked_patients.csv")

print("=== INTERLINKED PATIENT RESULTS ===")
print(f"Total patients in database: {len(df):,}")
print(f"Patients with CVD diagnosis: {df['has_cvd_diagnosis'].sum():,}")
print(f"Patients with X-rays: {df['has_xray'].sum():,}")
print(f"Patients with ECG: {df['has_ecg'].sum():,}")
print(f"Patients with ALL THREE: {df['has_all_three'].sum():,}")
print(f"Patients with Cardiomegaly: {df['has_cardiomegaly'].sum():,}")

interlinked = df[df['has_all_three']]
print(f"\nInterlinked with Cardiomegaly: {interlinked['has_cardiomegaly'].sum()}")

# Check IDs file
ids_file = data_dir / "interlinked_patient_ids.txt"
if ids_file.exists():
    with open(ids_file) as f:
        ids = f.readlines()
    print(f"\nInterlinked IDs saved: {len(ids)} patients")
