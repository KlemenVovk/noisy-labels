import torch
import torch.nn as nn
import torch.nn.functional as F


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, lmbd=3, beta=0.7):
        """Early Learning Regularization.
        Parameters
        * `num_examp` Total number of training examples.
        * `num_classes` Number of classes in the classification problem.
        * `lambda` Regularization strength; must be a positive float, controling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = (
            torch.zeros(num_examp, self.num_classes).cuda()
            if self.USE_CUDA
            else torch.zeros(num_examp, self.num_classes)
        )
        self.beta = beta
        self.lmbd = lmbd

    def forward(self, index, output, label):
        """Early Learning Regularization.
        Args
        * `index` Training sample index, due to training set shuffling,
        index is used to track training examples in different iterations.
        * `output` Model's logits, same as PyTorch provided loss functions.
        * `label` Labels, same as PyTorch provided loss functions.
        """

        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (
            (y_pred_) / (y_pred_).sum(dim=1, keepdim=True)
        )
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lmbd * elr_reg
        return final_loss
