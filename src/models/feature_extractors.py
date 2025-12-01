import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    """Conv1d with left-padding for causality."""
    def __init__(self, in_ch, out_ch, k, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=0)
        self.pad = k - 1
    
    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)

class XiHead(nn.Module):
    """Small projection head to re-embed waveforms back into CC representation spaces."""
    def __init__(self, sr=16000, kind='short', hidden=512, out_dim=64):
        super().__init__()
        # light conv stack + linear to 64-D
        if kind == 'short':
            downs = [5,4,2,2,2]
            kernels = [10,8,4,4,4]
        else:
            downs = [2,2,2]
            kernels = [4,4,4]
        ch = 1
        layers = []
        for k,s in zip(kernels, downs):
            layers += [CausalConv1d(ch, hidden, k, stride=s), nn.ReLU()]
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, wav):
        h = self.net(wav)        # [B, H, T']
        h = h.transpose(1, 2)    # [B, T', H]
        return self.proj(h)      # [B, T', 64]
