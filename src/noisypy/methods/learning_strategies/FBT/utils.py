import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


def estimate_noise_mtx(X_prob: Tensor, filter_outlier: bool = False, row_normalise: bool = True, clip_to_zero: bool = False):
        N, C = X_prob.shape
        T = torch.empty((C, C))

        # predict probability on the fresh sample
        eta_corr = X_prob

        # find a 'perfect example' for each class
        for i in range(C):

            if not filter_outlier:
                idx_best = torch.argmax(eta_corr[:, i])
            else:
                eta_thresh = torch.quantile(eta_corr[:, i], 97,
                                           interpolation='higher')
                robust_eta = eta_corr[:, i]
                robust_eta[robust_eta >= eta_thresh] = 0.0
                idx_best = torch.argmax(robust_eta)

            for j in range(C):
                T[i, j] = eta_corr[idx_best, j]
        
        if clip_to_zero:
            T[T < 1e-6] = 0

        if row_normalise:
            T /= T.sum(axis=-1, keepdim=True)

        return T


class ForwardTLoss(nn.Module):

    def __init__(self, T: torch.Tensor) -> None:
        super().__init__()
        self.T = nn.Parameter(T, requires_grad=False)
        self.n_classes = self.T.shape[0]

    def forward(self, y_pred, y_true):
        y_true = F.one_hot(y_true, self.n_classes).float()
        y_pred = F.softmax(y_pred, dim=-1)

        y_pred = y_pred / y_pred.sum(axis=-1, keepdims=True)
        y_pred = torch.clip(y_pred, min=1e-11, max=1 - 1e-11)
        #print(y_pred, torch.matmul(y_pred, self.T), sep="\n")
        return -(y_true * torch.log(torch.matmul(y_pred, self.T))).sum(axis=-1).mean()
    

class BackwardTLoss(nn.Module):

    def __init__(self, T: torch.Tensor) -> None:
        super().__init__()
        self.T_inv = nn.Parameter(torch.linalg.inv(T), requires_grad=False)
        self.n_classes = self.T_inv.shape[0]

    def forward(self, y_pred, y_true):
        y_true = F.one_hot(y_true, self.n_classes).float()
        y_pred = F.softmax(y_pred, dim=-1)

        y_pred = y_pred / y_pred.sum(axis=-1, keepdims=True)
        y_pred = torch.clip(y_pred, min=1e-11, max=1 - 1e-11)
        return -(torch.matmul(y_true, self.T_inv) * torch.log(y_pred)).sum(axis=-1).mean()
