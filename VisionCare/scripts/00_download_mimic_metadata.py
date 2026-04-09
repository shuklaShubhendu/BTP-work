"""
VisionCare - MIMIC Data Downloader (Pure Python)
=================================================
Downloads required MIMIC files using Python requests (no wget needed!).

Now uses MIMIC-IV Waveform (mimic4wdb) which matches MIMIC-IV/CXR patient IDs!
"""

import os
import sys
import urllib.request
import ssl
import base64
from pathlib import Path
from getpass import getpass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
try:
    from config import MIMIC_IV_DIR, MIMIC_CXR_DIR, MIMIC_WAVEFORM_DIR
    MIMIC_IV_DIR = Path(MIMIC_IV_DIR)
    MIMIC_CXR_DIR = Path(MIMIC_CXR_DIR)
    MIMIC_WAVEFORM_DIR = Path(MIMIC_WAVEFORM_DIR)
except ImportError:
    DATA_ROOT = Path("D:/VisionCare/data")
    MIMIC_IV_DIR = DATA_ROOT / "raw" / "mimic_iv"
    MIMIC_CXR_DIR = DATA_ROOT / "raw" / "mimic_cxr"
    MIMIC_WAVEFORM_DIR = DATA_ROOT / "raw" / "mimic_waveform"


# Files to download - NOW USING MIMIC-IV WAVEFORM (mimic4wdb)!
DOWNLOADS = [
    # MIMIC-IV v3.1
    ("patients.csv.gz", "https://physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz", MIMIC_IV_DIR),
    ("admissions.csv.gz", "https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz", MIMIC_IV_DIR),
    ("diagnoses_icd.csv.gz", "https://physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz", MIMIC_IV_DIR),
    
    # MIMIC-CXR v2.1.0
    ("mimic-cxr-2.0.0-metadata.csv.gz", "https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz", MIMIC_CXR_DIR),
    ("mimic-cxr-2.0.0-chexpert.csv.gz", "https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz", MIMIC_CXR_DIR),
    
    # MIMIC-IV Waveform v0.1.0 (NEW! matches MIMIC-IV patient IDs)
    ("RECORDS", "https://physionet.org/files/mimic4wdb/0.1.0/RECORDS", MIMIC_WAVEFORM_DIR),
]


def download_file(url: str, output_path: Path, username: str, password: str) -> bool:
    """Download a file with basic authentication."""
    print(f"  📥 {output_path.name}...", end=" ", flush=True)
    
    if output_path.exists():
        print("(skipped - already exists)")
        return True
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create auth header
        credentials = f"{username}:{password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        
        # Create request with auth
        request = urllib.request.Request(url)
        request.add_header("Authorization", f"Basic {encoded}")
        
        # SSL context
        context = ssl.create_default_context()
        
        # Download
        with urllib.request.urlopen(request, context=context) as response:
            data = response.read()
            
            with open(output_path, 'wb') as f:
                f.write(data)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ ({size_mb:.1f} MB)")
        return True
        
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("❌ Authentication failed!")
        elif e.code == 403:
            print("❌ Access denied! Check your dataset access.")
        elif e.code == 404:
            print("❌ File not found!")
        else:
            print(f"❌ HTTP Error {e.code}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("="*60)
    print("🫀 VisionCare - MIMIC Data Downloader")
    print("="*60)
    print("\n⚠️  NOW USING MIMIC-IV WAVEFORM (mimic4wdb v0.1.0)")
    print("   This matches MIMIC-IV/CXR patient IDs for 3-modality fusion!")
    
    print("\n📁 Download locations:")
    print(f"   MIMIC-IV:   {MIMIC_IV_DIR}")
    print(f"   MIMIC-CXR:  {MIMIC_CXR_DIR}")
    print(f"   Waveform:   {MIMIC_WAVEFORM_DIR}")
    
    print("\n🔐 Enter your PhysioNet credentials:")
    username = input("   Username: ").strip()
    password = getpass("   Password: ")
    
    print("\n📥 Downloading files...\n")
    
    success = 0
    failed = 0
    
    for filename, url, output_dir in DOWNLOADS:
        output_path = output_dir / filename
        if download_file(url, output_path, username, password):
            success += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print(f"📊 Results: {success} succeeded, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 All files downloaded!")
        print("\nNext step:")
        print("   python scripts/01_find_interlinked_patients.py")
    else:
        print("\n⚠️  Some downloads failed.")
        print("\n📋 Manual download links (login required):")
        for filename, url, output_dir in DOWNLOADS:
            output_path = output_dir / filename
            if not output_path.exists():
                print(f"   {url}")
                print(f"   → Save to: {output_dir}")


if __name__ == "__main__":
    main()
