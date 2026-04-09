"""
VisionCare - Step 2: Download Patient Data
==========================================
This script downloads X-ray images and ECG waveforms for the interlinked patients.
Run this AFTER 01_find_interlinked_patients.py

Note: You need PhysioNet credentials to download.
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration (paths point to D: drive)
try:
    from config import DATA_ROOT, MIMIC_CXR_DIR, MIMIC_WAVEFORM_DIR
    DATA_DIR = Path(DATA_ROOT)
    XRAY_OUTPUT_DIR = Path(MIMIC_CXR_DIR) / "images"
    ECG_OUTPUT_DIR = Path(MIMIC_WAVEFORM_DIR) / "waveforms"
except ImportError:
    print("⚠️  config.py not found, using default paths")
    DATA_DIR = Path("D:/VisionCare/data")
    XRAY_OUTPUT_DIR = DATA_DIR / "raw" / "mimic_cxr" / "images"
    ECG_OUTPUT_DIR = DATA_DIR / "raw" / "mimic_waveform" / "waveforms"

# Input files
INTERLINKED_IDS_FILE = DATA_DIR / "interlinked_patient_ids.txt"

# PhysioNet URLs
PHYSIONET_CXR_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/files"
PHYSIONET_WAVEFORM_URL = "https://physionet.org/files/mimic3wdb-matched/1.0"


def load_interlinked_patients():
    """Load the list of interlinked patient IDs."""
    if not INTERLINKED_IDS_FILE.exists():
        print(f"❌ {INTERLINKED_IDS_FILE} not found!")
        print("   Please run 01_find_interlinked_patients.py first.")
        sys.exit(1)
    
    with open(INTERLINKED_IDS_FILE, 'r') as f:
        patient_ids = [int(line.strip()) for line in f if line.strip()]
    
    print(f"✅ Loaded {len(patient_ids)} interlinked patient IDs")
    return patient_ids


def generate_xray_download_commands(patient_ids, metadata_path=None):
    """
    Generate wget commands to download X-rays for specific patients.
    
    X-ray images are organized as:
    files/p{XX}/p{XXXXXXXX}/s{XXXXXXXX}/{dicom_id}.jpg
    
    Where:
    - pXX is first 2 digits of subject_id (padded)
    - pXXXXXXXX is full subject_id (8 digits, padded)
    """
    print("\n🩻 Generating X-ray download commands...")
    
    commands = []
    
    for pid in tqdm(patient_ids, desc="   Building X-ray URLs"):
        # Format patient ID paths
        pid_str = str(pid).zfill(8)  # e.g., "00010006"
        prefix = f"p{pid_str[:2]}"   # e.g., "p00"
        patient_folder = f"p{pid_str}"  # e.g., "p00010006"
        
        # Download entire patient folder
        url = f"{PHYSIONET_CXR_URL}/{prefix}/{patient_folder}/"
        output_dir = XRAY_OUTPUT_DIR / prefix / patient_folder
        
        cmd = f'wget -r -N -c -np -nH --cut-dirs=4 -P "{output_dir}" --user YOUR_USERNAME --ask-password "{url}"'
        commands.append(cmd)
    
    return commands


def generate_ecg_download_commands(patient_ids, records_path=None):
    """
    Generate wget commands to download ECG waveforms for specific patients.
    
    Waveforms are organized as:
    p{XX}/p{XXXXXXX}/{record_name}.*
    """
    print("\n📈 Generating ECG download commands...")
    
    # Build patient folder paths
    patient_folders = {}
    for pid in patient_ids:
        pid_str = str(pid).zfill(7)  # e.g., "0000033"
        prefix = f"p{pid_str[:2]}"   # e.g., "p00"
        patient_folder = f"p{pid_str}"  # e.g., "p0000033"
        patient_folders[pid] = (prefix, patient_folder)
    
    commands = []
    
    for pid, (prefix, patient_folder) in tqdm(patient_folders.items(), desc="   Building ECG URLs"):
        url = f"{PHYSIONET_WAVEFORM_URL}/{prefix}/{patient_folder}/"
        output_dir = ECG_OUTPUT_DIR / prefix / patient_folder
        
        cmd = f'wget -r -N -c -np -nH --cut-dirs=3 -P "{output_dir}" --user YOUR_USERNAME --ask-password "{url}"'
        commands.append(cmd)
    
    return commands


def create_download_script(xray_commands, ecg_commands):
    """Create a bash/batch script for downloading."""
    
    # Create output directories
    XRAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ECG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Bash script for Linux/Mac/WSL
    bash_script = DATA_DIR / "download_data.sh"
    with open(bash_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# VisionCare Data Download Script\n")
        f.write("# Replace YOUR_USERNAME with your PhysioNet username\n\n")
        
        f.write("echo '=== Downloading X-ray Images ==='\n")
        for i, cmd in enumerate(xray_commands[:10], 1):  # Limit to 10 for demo
            f.write(f"echo 'Downloading patient {i}/{len(xray_commands[:10])}'\n")
            f.write(cmd + "\n")
        
        f.write("\necho '=== Downloading ECG Waveforms ==='\n")
        for i, cmd in enumerate(ecg_commands[:10], 1):  # Limit to 10 for demo
            f.write(f"echo 'Downloading patient {i}/{len(ecg_commands[:10])}'\n")
            f.write(cmd + "\n")
    
    # Full commands file (for all patients)
    full_commands_file = DATA_DIR / "all_download_commands.txt"
    with open(full_commands_file, 'w') as f:
        f.write("# X-RAY DOWNLOAD COMMANDS\n")
        f.write("# ========================\n")
        for cmd in xray_commands:
            f.write(cmd + "\n")
        
        f.write("\n# ECG DOWNLOAD COMMANDS\n")
        f.write("# =====================\n")
        for cmd in ecg_commands:
            f.write(cmd + "\n")
    
    print(f"\n✅ Created download scripts:")
    print(f"   - Sample script (10 patients): {bash_script}")
    print(f"   - Full commands list: {full_commands_file}")
    
    return bash_script, full_commands_file


def print_manual_instructions(patient_ids):
    """Print manual download instructions for the user."""
    
    print("\n" + "="*70)
    print("📥 DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    print("""
Option 1: Use the PhysioNet Web Interface (Easiest)
----------------------------------------------------
1. Go to https://physionet.org/content/mimic-cxr-jpg/2.0.0/
2. Click 'Files' tab
3. Navigate to 'files/' folder
4. Download folders for your interlinked patients

Option 2: Use wget (Recommended for many patients)
--------------------------------------------------
1. Open a terminal (WSL on Windows, or Git Bash)
2. Run the commands in: data/download_data.sh
3. You'll be prompted for your PhysioNet password

Option 3: Use Google Cloud (Fastest for full dataset)
------------------------------------------------------
MIMIC data is available on Google Cloud:
- mimic-cxr-jpg: gs://mimic-cxr-jpg-2.0.0.physionet.org
- mimic3wdb: gs://mimic3wdb-matched-1.0.physionet.org

Use gsutil to download specific patient folders.
""")
    
    print(f"\nPatients to download: {len(patient_ids)}")
    print(f"Estimated X-ray data: ~{len(patient_ids) * 30}MB - {len(patient_ids) * 50}MB")
    print(f"Estimated ECG data: ~{len(patient_ids) * 5}MB - {len(patient_ids) * 20}MB")
    print("="*70)


def main():
    print("="*60)
    print("🫀 VisionCare - Download Patient Data")
    print("="*60)
    
    # Load interlinked patients
    patient_ids = load_interlinked_patients()
    
    # Generate download commands
    xray_commands = generate_xray_download_commands(patient_ids)
    ecg_commands = generate_ecg_download_commands(patient_ids)
    
    # Create download scripts
    create_download_script(xray_commands, ecg_commands)
    
    # Print manual instructions
    print_manual_instructions(patient_ids)
    
    print("\n✅ COMPLETE! Review the download scripts and instructions above.")


if __name__ == "__main__":
    main()
