# src/ai/model_pose.py
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Two model options:
 - PoseLSTM: simple LSTM classifier (default)
 - PoseTCN: small Temporal ConvNet (optional)
Both expect input: (batch, seq_len, feat_dim)
Outputs: logits for classification (num_classes)
"""

class PoseLSTM(nn.Module):
    def __init__(self, feat_dim, hidden_size=256, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(feat_dim, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)  # out: (B, T, hidden)
        last = out[:, -1, :]  # take last timestep
        logits = self.fc(last)
        return logits

# Optional: small TCN block
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class PoseTCN(nn.Module):
    def __init__(self, feat_dim, num_channels=[128, 128], kernel_size=3, num_classes=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = feat_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(num_channels[-1], 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        # x: (B, T, F) -> TCN expects (B, F, T)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        # y: (B, C, T) -> pool over time
        y = y.mean(dim=2)
        logits = self.fc(y)
        return logits