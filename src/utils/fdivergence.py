import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision import transforms

# TODO: in their code they have the dataloaders and transforms for noisy/not noisy, train/val/test, regular/peer, ... Think how to generalize this spaghetti code...
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_cifar10_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
])
# TODO: should we have a val transform or not? In their code they transform the val and test data in the same way
val_cifar10_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
])
test_cifar10_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
])


class CrossEntropyLossStable(nn.Module):
    def __init__(self, reduction="mean", eps=1e-5):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss(torch.log(self._softmax(outputs) + self._eps), labels)


class ProbLossStable(nn.Module):
    def __init__(self, reduction="none", eps=1e-5):
        super(ProbLossStable, self).__init__()
        self._name = "Prob Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction="none")

    def forward(self, outputs, labels):
        return self._nllloss(self._softmax(outputs), labels)


class Divergence():
    def __init__(self, name="Total-Variation"):
        self.name = name
        self._act_conj_func_assign(name)

    def _act_conj_func_assign(self, name):        
        match name:
            case "KL":
                self.activation = lambda x: -torch.mean(x)
                self.conjugate = lambda x: -torch.mean(torch.exp(x - 1.))
            case "Reverse-KL":
                self.activation = lambda x: -torch.mean(-torch.exp(x))
                self.conjugate = lambda x: -torch.mean(-1. - x)  # remove log
            case "Jeffrey":
                self.activation = lambda x: -torch.mean(x)
                self.conjugate = lambda x: -torch.mean(x + torch.mul(x, x) / 4. + torch.mul(torch.mul(x, x), x) / 16.)
            case "Squared-Hellinger":
                self.activation = lambda x: -torch.mean(1. - torch.exp(x))
                self.conjugate = lambda x: -torch.mean((1. - torch.exp(x)) / (torch.exp(x)))
            case "Pearson":
                self.activation = lambda x: -torch.mean(x)
                self.conjugate = lambda x: -torch.mean(torch.mul(x, x) / 4. + x)
            case "Neyman":
                self.activation = lambda x: -torch.mean(1. - torch.exp(x))
                self.conjugate = lambda x: -torch.mean(2. - 2. * torch.sqrt(1. - x))
            case "Jenson-Shannon":
                self.activation = lambda x: -torch.mean(- torch.log(1. + torch.exp(-x))) - torch.log(torch.tensor(2.))
                self.conjugate = lambda x:  -torch.mean(x + torch.log(1. + torch.exp(-x))) + torch.log(torch.tensor(2.))
            case "Total-Variation":
                self.activation = lambda x: -torch.mean(torch.tanh(x) / 2.)
                self.conjugate = lambda x: -torch.mean(torch.tanh(x) / 2.)
            case _:
                raise ValueError("Divergence name must be in ['KL', 'Reverse-KL', 'Jeffrey', 'Squared-Hellinger', 'Pearson', 'Neyman', 'Jenson-Shannon', 'Total-Variation'].\nProvided divergence '{name}' not implemented.")
    
    def activation(self, x):
        return self.activation(x)
    
    def conjugate(self, x):
        return self.conjugate(x)
