import torch
from torch import nn
from torch.nn import functional as F

# optimized code, no numpy operations, no explicit to.(cuda) usage
# NOTE: not numerically stable - currently, lr = 0.1 NaNs out gradients


class GCELoss(nn.Module):
    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super().__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(
            data=torch.ones(trainset_size, 1), requires_grad=False
        )

    def forward(self, logits, targets, indexes):
        B = logits.shape[0]
        p = F.softmax(logits, dim=-1)
        Yg = p[torch.arange(B), targets].unsqueeze(1)

        loss = ((1 - Yg**self.q) / self.q) * self.weight[indexes] - (
            (1 - self.k**self.q) / self.q
        ) * self.weight[indexes]
        return loss.mean()

    def update_weight(self, logits, targets, indexes):
        B = logits.shape[0]
        p = F.softmax(logits, dim=1)
        Yg = p[torch.arange(B), targets].unsqueeze(1)

        Lq = (1 - Yg**self.q) / self.q
        Lqk = (1 - self.k**self.q) / self.q

        self.weight[indexes] = (Lqk > Lq).float()
