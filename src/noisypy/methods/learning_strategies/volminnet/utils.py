import torch
from torch import nn
from torch.nn import functional as F


class SigT(nn.Module):
    def __init__(self, num_classes, init=2):
        super().__init__()
        self.w = nn.Parameter(-init * torch.ones(num_classes, num_classes))
        self.register_buffer("eye", torch.eye(num_classes))

    def forward(self):
        T = self.eye + (1 - self.eye) * F.sigmoid(self.w)
        return F.normalize(T, p=1, dim=1)
