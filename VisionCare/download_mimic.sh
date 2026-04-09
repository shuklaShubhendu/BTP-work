#!/bin/bash
echo '============================================'
echo 'VisionCare - MIMIC Data Downloader'
echo '============================================'
echo ''
echo 'You will be prompted for your PhysioNet password for each file.'
echo ''

echo 'Downloading MIMIC-IV patients.csv.gz...'
wget -c -N --user shubhendu9966 --ask-password -O "D:/VisionCare/data/raw/mimic_iv/patients.csv.gz" https://physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz
echo ''

echo 'Downloading MIMIC-IV admissions.csv.gz...'
wget -c -N --user shubhendu9966 --ask-password -O "D:/VisionCare/data/raw/mimic_iv/admissions.csv.gz" https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz
echo ''

echo 'Downloading MIMIC-IV diagnoses_icd.csv.gz...'
wget -c -N --user shubhendu9966 --ask-password -O "D:/VisionCare/data/raw/mimic_iv/diagnoses_icd.csv.gz" https://physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz
echo ''

echo 'Downloading MIMIC-IV d_icd_diagnoses.csv.gz...'
wget -c -N --user shubhendu9966 --ask-password -O "D:/VisionCare/data/raw/mimic_iv/d_icd_diagnoses.csv.gz" https://physionet.org/files/mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz
echo ''

echo 'Downloading MIMIC-CXR metadata...'
wget -c -N --user shubhendu9966 --ask-password -O "D:/VisionCare/data/raw/mimic_cxr/mimic-cxr-2.0.0-metadata.csv.gz" https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz
echo ''

echo 'Downloading MIMIC-CXR chexpert labels...'
wget -c -N --user shubhendu9966 --ask-password -O "D:/VisionCare/data/raw/mimic_cxr/mimic-cxr-2.0.0-chexpert.csv.gz" https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz
echo ''

echo 'Downloading MIMIC Waveform RECORDS...'
wget -c -N --user shubhendu9966 --ask-password -O "D:/VisionCare/data/raw/mimic_waveform/RECORDS" https://physionet.org/files/mimic3wdb-matched/1.0/RECORDS
echo ''

echo '============================================'
echo 'Download complete!'
echo '============================================'
