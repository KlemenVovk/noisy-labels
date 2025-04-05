import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.beta import Beta
from torchvision import transforms

from noisypy.data.datasets.cifar10 import cifar10_train_transform


weak_train_transform = cifar10_train_transform

strong_train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class GCELoss(nn.Module):
    def __init__(self, num_classes, q=0.7, gpu=None):
        super(GCELoss, self).__init__()
        self.device = torch.device("cuda:%s" % gpu) if gpu else torch.device("cpu")
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = (
            torch.nn.functional.one_hot(labels, self.num_classes)
            .float()
            .to(self.device)
        )
        gce = (1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()


class Mixup(nn.Module):
    def __init__(self, gpu=None, num_classes=10, alpha=5.0, model=None):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda:%s" % gpu) if gpu else torch.device("cpu")
        self.alpha = torch.tensor(alpha).to(self.device)
        if model:
            model.dummy_head = nn.Linear(512, num_classes, bias=True).to(self.device)
            model.fc.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, data_in, data_out):
        self.features = data_in[0]

    def forward(self, x, y, model):
        b = x.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        lam = max(lam, 1 - lam)
        index = torch.randperm(b).to(self.device)
        y = (
            torch.zeros(b, self.num_classes)
            .to(self.device)
            .scatter_(1, y.view(-1, 1), 1)
        )
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        mixed_p = model(mixed_x)
        loss = -torch.mean(torch.sum(F.log_softmax(mixed_p, dim=1) * mixed_y, dim=1))
        return loss

    def ws_forward(self, wx, sx, y, model):
        b = wx.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        lam = max(lam, 1 - lam)
        index = torch.randperm(b).to(self.device)
        y = (
            torch.zeros(b, self.num_classes)
            .to(self.device)
            .scatter_(1, y.view(-1, 1), 1)
        )
        mixed_x = lam * wx + (1 - lam) * sx[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        mixed_p = model(mixed_x)
        loss = -torch.mean(torch.sum(F.log_softmax(mixed_p, dim=1) * mixed_y, dim=1))
        return loss

    def soft_forward(self, x, p, model):
        b = x.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        index = torch.randperm(b).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * p + (1 - lam) * p[index]
        mixed_p = model(mixed_x)
        loss = -torch.mean(torch.sum(F.log_softmax(mixed_p, dim=1) * mixed_y, dim=1))
        return loss

    def dummy_forward(self, x, y, model):
        b = x.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        lam = max(lam, 1 - lam)
        index = torch.randperm(b).to(self.device)
        y = (
            torch.zeros(b, self.num_classes)
            .to(self.device)
            .scatter_(1, y.view(-1, 1), 1)
        )
        mixed_y = lam * y + (1 - lam) * y[index]
        mixed_features = self.features.clone()
        dummy_logits = model.dummy_head(mixed_features)
        loss = -torch.mean(
            torch.sum(F.log_softmax(dummy_logits, dim=1) * mixed_y, dim=1)
        )
        return loss
