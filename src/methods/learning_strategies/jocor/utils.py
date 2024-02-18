import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.nn.functional as F

# TODO: remove numpy and .cpu() calls

#def kl_loss_compute(pred, soft_targets, reduce=True):
#    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)
#    if reduce:
#        return torch.mean(torch.sum(kl, dim=1))
#    else:
#        return torch.sum(kl, 1)

def kl_loss_compute(pred, soft_targets, reduce=True):
    kl = F.kl_div(
        F.log_softmax(pred, dim=1),
        F.log_softmax(soft_targets, dim=1),
        reduction="none", log_target=True
    )
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def loss_jocor(y_1, y_2, tgt, forget_rate, co_lambda):
    loss_pick_1 = F.cross_entropy(y_1, tgt, reduction="none") * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, tgt, reduction="none") * (1-co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2, reduce=False) +\
        co_lambda * kl_loss_compute(y_2, y_1, reduce=False)).cpu()

    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))
    ind_update=ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])
    return loss


# TODO: remove

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel, 64,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(128,196,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1)
        self.linear1=nn.Linear(256, n_outputs)
        self.bn1=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn2=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn3=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn4=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn5=nn.BatchNorm2d(196, momentum=self.momentum)
        self.bn6=nn.BatchNorm2d(16, momentum=self.momentum)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        h=self.c2(h)
        h=F.relu(call_bn(self.bn2, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c3(h)
        h=F.relu(call_bn(self.bn3, h))
        h=self.c4(h)
        h=F.relu(call_bn(self.bn4, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c5(h)
        h=F.relu(call_bn(self.bn5, h))
        h=self.c6(h)
        h=F.relu(call_bn(self.bn6, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit=self.linear1(h)
        return logit