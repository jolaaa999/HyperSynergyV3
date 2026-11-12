import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.size()
        s = x.mean(dim=[2, 3])
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        s = s.view(b, c, 1, 1)
        return x * s


class CellSEConv(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.gconv1 = nn.Conv2d(1, 32, 7, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.gconv2 = nn.Conv2d(32, 64, 5, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gconv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.se3 = SE(128)
        self.gconv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.se4 = SE(64)
        self.fc = nn.Linear(64 * 25, out_dim)

    def forward(self, cell):
        if cell.dim() == 3 and cell.size(1) == 1:
            cell = cell.squeeze(1)
        if cell.dim() == 2:
            spc, h = cell.size()
            cell = cell.view(spc, int(h ** 0.5), int(h ** 0.5)).unsqueeze(1)
        elif cell.dim() == 3 and cell.size(-2) == 30 and cell.size(-1) == 30:
            cell = cell.unsqueeze(1)
        x = self.pool1(F.relu(self.norm1(self.gconv1(cell))))
        x = self.pool2(F.relu(self.norm2(self.gconv2(x))))
        x = F.relu(self.gconv3(x))
        x = self.se3(x)
        x = F.relu(self.gconv4(x))
        x = self.se4(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h * w)
        return self.fc(x)
