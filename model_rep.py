import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import SampleFeatureExtractor


class EMBNet(nn.Module):
    def __init__(self, n_output=1, dropout=0.3, cell_encoder_config=None):
        super().__init__()
        # [Innovation-1: Biological Graph Cell Encoder]
        self.extractor = SampleFeatureExtractor(cell_encoder_config=cell_encoder_config)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.out = nn.Linear(1024, n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_a, drug_b):
        x = self.extractor(drug_a, drug_b)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.out(x)
