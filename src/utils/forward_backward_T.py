import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class ForwardT(nn.Module):

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
    

class BackwardT(nn.Module):

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

'''
# unchanged from original implementation
# estimate_noise_mtx replaces these two classes
# tests show same results
class NoiseEstimator():

    def __init__(self, classifier, row_normalize=True, alpha=0.0,
                 filter_outlier=False, cliptozero=False, verbose=0):
        """classifier: an ALREADY TRAINED model. In the ideal case, classifier
        should be powerful enough to only make mistakes due to label noise."""

        self.classifier = classifier
        self.row_normalize = row_normalize
        self.alpha = alpha
        self.filter_outlier = filter_outlier
        self.cliptozero = cliptozero
        self.verbose = verbose

    def fit(self, X):

        # number of classes
        c = self.classifier.classes
        T = np.empty((c, c))

        # predict probability on the fresh sample
        eta_corr = self.classifier.predict_proba(X)

        # find a 'perfect example' for each class
        for i in np.arange(c):

            if not self.filter_outlier:
                idx_best = np.argmax(eta_corr[:, i])
            else:
                eta_thresh = np.percentile(eta_corr[:, i], 97,
                                           interpolation='higher')
                robust_eta = eta_corr[:, i]
                robust_eta[robust_eta >= eta_thresh] = 0.0
                idx_best = np.argmax(robust_eta)

            for j in np.arange(c):
                T[i, j] = eta_corr[idx_best, j]

        self.T = T
        return self

    def predict(self):

        T = self.T
        c = self.classifier.classes

        if self.cliptozero:
            idx = np.array(T < 10 ** -6)
            T[idx] = 0.0

        if self.row_normalize:
            row_sums = T.sum(axis=1)
            T /= row_sums[:, np.newaxis]

        if self.verbose > 0:
            print(T)

        if self.alpha > 0.0:
            T = self.alpha * np.eye(c) + (1.0 - self.alpha) * T

        if self.verbose > 0:
            print(T)
            print(np.linalg.inv(T))

        return T
'''
    
    
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
