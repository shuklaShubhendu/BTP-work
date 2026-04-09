"""
VisionCare - Setup Script
=========================
Run this script FIRST to create the data directory structure on D: drive.
"""

import os
from pathlib import Path

# Import configuration
try:
    from config import (
        DATA_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        MIMIC_IV_DIR, MIMIC_CXR_DIR, MIMIC_WAVEFORM_DIR,
        MODEL_CHECKPOINT_DIR
    )
except ImportError:
    # Fallback defaults
    DATA_ROOT = "D:/VisionCare/data"
    RAW_DATA_DIR = f"{DATA_ROOT}/raw"
    PROCESSED_DATA_DIR = f"{DATA_ROOT}/processed"
    MIMIC_IV_DIR = f"{RAW_DATA_DIR}/mimic_iv"
    MIMIC_CXR_DIR = f"{RAW_DATA_DIR}/mimic_cxr"
    MIMIC_WAVEFORM_DIR = f"{RAW_DATA_DIR}/mimic_waveform"
    MODEL_CHECKPOINT_DIR = f"{DATA_ROOT}/checkpoints"


def create_directory_structure():
    """Create all required directories on D: drive."""
    
    directories = [
        DATA_ROOT,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MIMIC_IV_DIR,
        MIMIC_CXR_DIR,
        MIMIC_WAVEFORM_DIR,
        MODEL_CHECKPOINT_DIR,
        f"{MIMIC_CXR_DIR}/images",
        f"{MIMIC_WAVEFORM_DIR}/waveforms",
        f"{PROCESSED_DATA_DIR}/xray_tensors",
        f"{PROCESSED_DATA_DIR}/ecg_tensors",
        f"{PROCESSED_DATA_DIR}/clinical_features",
    ]
    
    print("="*60)
    print("🫀 VisionCare - Creating Directory Structure")
    print("="*60)
    print(f"\nData root: {DATA_ROOT}")
    print("\nCreating directories:")
    
    for directory in directories:
        path = Path(directory)
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {path}")
        except Exception as e:
            print(f"  ❌ {path} - Error: {e}")
    
    print("\n" + "="*60)
    print("📁 Directory Structure Created!")
    print("="*60)
    
    # Print instructions
    print("""
Next Steps:
-----------
1. Download MIMIC metadata files from PhysioNet

2. Place files in the following locations:

   MIMIC-IV files → """ + MIMIC_IV_DIR + """
   - patients.csv.gz
   - admissions.csv.gz  
   - diagnoses_icd.csv.gz

   MIMIC-CXR files → """ + MIMIC_CXR_DIR + """
   - mimic-cxr-2.0.0-metadata.csv.gz
   - mimic-cxr-2.0.0-chexpert.csv

   MIMIC Waveform files → """ + MIMIC_WAVEFORM_DIR + """
   - RECORDS (from mimic3wdb-matched)

3. Run: python scripts/01_find_interlinked_patients.py
""")
    
    # Check available disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(DATA_ROOT)
        print(f"\n💾 Disk Space on {DATA_ROOT[:2]}:")
        print(f"   Total: {total // (1024**3)} GB")
        print(f"   Free:  {free // (1024**3)} GB")
        print(f"   Needed: ~50 GB (estimated)")
    except Exception:
        pass


if __name__ == "__main__":
    create_directory_structure()
