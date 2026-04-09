# ============================================================
# PART 1: SETUP & DATA DOWNLOAD
# Run this cell first in Colab
# ============================================================

import os
from google.colab import drive

# Mount Drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# Install dependencies
!pip install -q torch torchvision torchaudio
!pip install -q wfdb scikit-learn matplotlib seaborn tqdm

# AWS CLI for download
!pip install -q awscli

# Define paths
DEST_DIR = "/content/drive/MyDrive/symile-mimic/"
os.makedirs(DEST_DIR, exist_ok=True)

print(f"📁 Destination: {DEST_DIR}")

# ============================================================
# DOWNLOAD SYMILE-MIMIC DATASET
# ============================================================

source_uri = "s3://arn:aws:s3:us-east-1:724665945834:accesspoint/symile-mimic-v1-0-0-01/symile-mimic/1.0.0/"

print("🚀 Downloading Symile-MIMIC dataset...")
print("⏱️ This takes 15-30 minutes depending on connection...")

!aws s3 sync "{source_uri}" "{DEST_DIR}" --request-payer requester --no-sign-request

print("✅ Download Complete!")
!ls -la {DEST_DIR}
