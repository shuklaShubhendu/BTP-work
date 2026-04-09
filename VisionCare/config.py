# VisionCare Configuration
# ========================
# Edit this file to set your data paths

# Where to store MIMIC data (change this to your D: drive path!)
DATA_ROOT = "D:/VisionCare/data"

# Alternatively, use raw string for Windows paths:
# DATA_ROOT = r"D:\VisionCare\data"

# Sub-directories (will be created automatically)
RAW_DATA_DIR = f"{DATA_ROOT}/raw"
PROCESSED_DATA_DIR = f"{DATA_ROOT}/processed"

# MIMIC dataset paths
MIMIC_IV_DIR = f"{RAW_DATA_DIR}/mimic_iv"
MIMIC_CXR_DIR = f"{RAW_DATA_DIR}/mimic_cxr"
MIMIC_WAVEFORM_DIR = f"{RAW_DATA_DIR}/mimic_waveform"

# Model checkpoints
MODEL_CHECKPOINT_DIR = f"{DATA_ROOT}/checkpoints"

# PhysioNet credentials (optional - can use env vars instead)
# PHYSIONET_USERNAME = "your_username"
