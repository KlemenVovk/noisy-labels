import torch
from torch import nn


# TODO: try normal crossentropy
class CrossEntropyLossStable(nn.Module):
    def __init__(self, reduction='mean', eps=1e-5):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels )


# TODO: rewrite so it's not hardcoded to max of 340 epochs
def f_alpha(epoch, r=0.1):
    if r <= 0.3:
    # Sparse setting
        alpha1 = torch.linspace(0.0, 0.0, 20)
        alpha2 = torch.linspace(0.0, 1, 20)
        alpha3 = torch.linspace(1, 2, 50)
        alpha4 = torch.linspace(2, 5, 50)
        alpha5 = torch.linspace(5, 10, 100)
        alpha6 = torch.linspace(10, 20, 100)
    else:
    # Uniform/Random noise setting
        alpha1 = torch.linspace(0.0, 0.0, 20)
        alpha2 = torch.linspace(0.0, 0.1, 20)
        alpha3 = torch.linspace(1, 2, 50)
        alpha4 = torch.linspace(2, 2.5, 50)
        alpha5 = torch.linspace(2.5, 3.3, 100)
        alpha6 = torch.linspace(3.3, 5, 100)
    alpha = torch.concatenate((alpha1, alpha2, alpha3, alpha4, alpha5, alpha6),axis=0)
    epoch = max(0, min(epoch, 339)) # clamp so it's in range
    return alpha[epoch]

def lr_plan(epoch):
    epoch = max(0, min(epoch, 319))
    lr_list = [0.1] * 40 + [0.01] * 40 + [0.001] * 40 + [1e-4] * 40 + [1e-5] * 40 + [1e-6] * 40 +  [1e-7] * 40 +  [1e-8] * 20
    return lr_list[epoch] / (1 + f_alpha(epoch))

