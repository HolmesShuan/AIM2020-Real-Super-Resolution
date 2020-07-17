from model import common
import torch
import torch.nn as nn

class ClipL1(nn.Module):
    def __init__(self, clip_min=0.0, clip_max=10.0):
        super(ClipL1, self).__init__()
        self.clip_max = clip_max
        self.clip_min = clip_min

    def forward(self, sr, hr):
        loss = torch.mean(torch.clamp(torch.abs(sr-hr), self.clip_min, self.clip_max))
        return loss
