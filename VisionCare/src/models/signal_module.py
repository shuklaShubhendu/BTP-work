"""
Signal Module - 1D-CNN and LSTM for ECG Analysis
VisionCare: Cardiovascular Disease Detection
Project: BTP Semester 7
"""

import torch
import torch.nn as nn


class ECG1DCNN(nn.Module):
    """1D-CNN for 12-lead ECG. Input: (B, 12, 5000), Output: (B, 2), (B, 256)"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.name = "1D-CNN"
        self.feature_dim = 256
        
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 15, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.2),
            nn.Conv1d(64, 128, 11, padding=5), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.2),
            nn.Conv1d(128, 256, 7, padding=3), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))
    
    def forward(self, x):
        features = self.conv(x).squeeze(-1)
        return self.classifier(features), features


class ECGLSTM(nn.Module):
    """BiLSTM for ECG. Input: (B, 12, 5000), Output: (B, 2), (B, 256)"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.name = "BiLSTM"
        self.feature_dim = 256
        
        self.downsample = nn.Sequential(nn.Conv1d(12, 64, 15, stride=5, padding=7), nn.BatchNorm1d(64), nn.ReLU())
        self.lstm = nn.LSTM(64, 128, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(256, 256)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))
    
    def forward(self, x):
        x = self.downsample(x).transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        features = self.fc(torch.cat([h_n[-2], h_n[-1]], dim=1))
        return self.classifier(features), features


def get_signal_model(name='cnn', num_classes=2):
    return {'cnn': ECG1DCNN, 'lstm': ECGLSTM}[name](num_classes)
