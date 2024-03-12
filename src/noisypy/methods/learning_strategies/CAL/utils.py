import torch
import torch.nn as nn

# ======================================================================== #
# Peer loss with fixed peer sample size = 1
# ======================================================================== #
class CrossEntropyLossStable(nn.Module):
    '''
    For use in PeerLossOne, as the original CrossEntropyLoss are likely be
    blowup when using in the peer term
    '''
    def __init__(self, reduction='mean', eps=1e-8):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels)


class CrossEntropyLossRegStable(nn.Module):
    def __init__(self, noisy_prior, eps=1e-5): 
        # eps = 1e-5 for most settings
        # try to find the eps for mixup settings
        super(CrossEntropyLossRegStable, self).__init__()
        self._prior = noisy_prior
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
    
    def forward(self, outputs, clusteridx = None):
        r'''
        labels are real vectors (from one-hot-encoding)
        ------
        outputs : batch * num_class
        labels  : batch * num_class
        '''
        log_out = torch.log( self._softmax(outputs) + self._eps )
        noise_prior = torch.zeros_like(outputs)
        res = torch.sum(torch.mul(self._prior, log_out), dim=1)
        return -torch.mean(res)


class CrossEntropyStableCALMultiple(nn.Module):
    def __init__(self, T_mat, T_mat_true,P_y_distill, eps=1e-5): 
        # eps = 1e-5 for most settings
        # try to find the eps for mixup settings
        super(CrossEntropyStableCALMultiple, self).__init__()
        self.T_mat = T_mat
        self.T_mat_true = T_mat_true
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self.P_y_distill = P_y_distill
    
    def forward(self, outputs, true_y, distill_y, raw_idx, loss_mean_y, loss_mean_n, loss_mean_all, loss_mean_y_true, loss_mean_n_true, loss_mean_all_true, distilled_weights):
        r'''
        labels are real vectors (from one-hot-encoding)
        ------
        outputs : batch * num_class
        labels  : batch * num_class
        true_y is actually the distilled y
        distilled_weights is beta
        '''
        log_out = -torch.log( self._softmax(outputs) + self._eps )
        T_mat = self.T_mat[:,:,raw_idx]
        T_mat_indicator = torch.sum(T_mat>0.0,1).view(T_mat.shape[0],1,-1).repeat(1,T_mat.shape[0],1).float()
        if self.T_mat_true is not None:
            T_mat_true = self.T_mat_true[:,:,raw_idx]
            T_mat_indicator_true = torch.sum(T_mat_true>0.0,1).view(T_mat_true.shape[0],1,-1).repeat(1,T_mat_true.shape[0],1).float()
        loss_all = torch.transpose(log_out,0,1).view(1,T_mat.shape[0],-1).repeat(T_mat.shape[0],1,1)
        # (torch.sum(T_mat_indicator * loss_all, dim = 2)/torch.sum(T_mat_indicator, dim = 2))
        loss_all_norm = (loss_all - loss_mean_all.view(T_mat.shape[0],T_mat.shape[0],1).repeat(1,1,T_mat.shape[2])) * T_mat_indicator
        if self.T_mat_true is not None:
            loss_all_norm_true = (loss_all - loss_mean_all.view(T_mat.shape[0],T_mat.shape[0],1).repeat(1,1,T_mat.shape[2])) * T_mat_indicator_true
            loss_rec_all_true = torch.sum(T_mat_indicator_true * loss_all, dim = 2)
        loss_rec_all = torch.sum(T_mat_indicator * loss_all, dim = 2)
        

        T_mat_indicator_sum = torch.sum(T_mat_indicator, dim = 2)
        T_mat_indicator_sum[T_mat_indicator_sum==0] = 1.0
        if self.T_mat_true is not None:
            T_mat_indicator_true_sum = torch.sum(T_mat_indicator_true, dim = 2)
            T_mat_indicator_true_sum[T_mat_indicator_true_sum==0] = 1.0
            loss_out_true = torch.sum(torch.sum(torch.sum(T_mat_true * loss_all_norm_true, dim = 2)/T_mat_indicator_true_sum, dim = 1) * 1.0/T_mat.shape[0])
        loss_out = torch.sum(torch.sum(torch.sum(T_mat * loss_all_norm, dim = 2)/T_mat_indicator_sum, dim = 1) * self.P_y_distill) # TODO: test even p_y_distill
        if self.T_mat_true is not None:
            return loss_out, loss_out_true, loss_rec_all, loss_rec_all_true
        else:
            return loss_out, torch.tensor(0.0), loss_rec_all, torch.tensor(0.0)


class CrossEntropyLossRegStableMix(nn.Module):
    def __init__(self, noisy_prior, eps=1e-2):  # 
        # eps = 1e-5 for most settings
        # try to find the eps for mixup settings
        super(CrossEntropyLossRegStableMix, self).__init__()
        # noisy_prior_0 = noisy_prior.repeat(noisy_prior.shape[0],1)
        # noisy_prior_1 = noisy_prior_0.transpose(0,1)
        # self._prior = noisy_prior_0 + noisy_prior_1
        self._prior = noisy_prior
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
    
    def forward(self, outputs, lmd, noise_prior_new = None, weight = None):
        r'''
        labels are real vectors (from one-hot-encoding)
        ------
        outputs : batch * num_class    mixup outputs
        labels  : batch * num_class
        y1, y2 are noisy labels
        \sum_{y1,y2} P(y1 or y2)*log(f_x(y1)+f_x(y2))
        '''   
        length = outputs.shape[0]//2
        # log_out = torch.log( torch.abs(self._softmax(outputs[:length]) - lmd) + self._eps )
        log_out = torch.log( self._softmax(outputs[length:]) + self._eps )
        if noise_prior_new is not None:
            res = torch.sum(torch.mul(noise_prior_new, log_out), dim=1)
        else:
            res = torch.sum(torch.mul(self._prior, log_out), dim=1)
        if weight is not None:
            weight_var = torch.tensor(weight).to(self._device)
            if sum(weight) == 0:
                return -torch.sum(res*weight_var)
            else:
                return -torch.sum(res*weight_var)/sum(weight)
        else:
            return -torch.mean(res)


        return -torch.mean(res)


class PeerLossRegCE(nn.Module):
    def __init__(self, alpha, noisy_prior, loss_name, T_mat = None, T_mat_true = None, P_y_distill = None):
        super(PeerLossRegCE, self).__init__()
        self._name = "peer loss function with noisy prior"
        self._lossname = loss_name
        if loss_name == 'crossentropy':
            self._peer = CrossEntropyLossRegStable(noisy_prior)
        elif loss_name == 'crossentropy_CAL':
            self._peer = CrossEntropyLossRegStable(noisy_prior)
            self._CAL = CrossEntropyStableCALMultiple(T_mat,T_mat_true,P_y_distill)

        self._ce = CrossEntropyLossStable()
        self._alpha = alpha if alpha is not None else 1.
    
    def forward(self, outputs, labels, output_peer, labels_nomix = None, lmd = None, noisy_prior_new = None, weight = None, true_y = None, distill_y = None, raw_idx = None, loss_mean_y = None, loss_mean_n = None, loss_mean_all = None, distilled_weights = None, loss_mean_y_true = None, loss_mean_n_true = None, loss_mean_all_true = None):
        # calculate the base loss
        base_loss = self._ce(outputs, labels)
        peer_term = self._peer(output_peer)
        if self._lossname == "crossentropy_CAL":
            CAL_term, CAL_term_true, loss_rec_all, loss_rec_all_true = self._CAL(outputs, true_y, distill_y, raw_idx, loss_mean_y, loss_mean_n, loss_mean_all, loss_mean_y_true, loss_mean_n_true, loss_mean_all_true, distilled_weights)
            return base_loss - self._alpha * peer_term - CAL_term, base_loss, peer_term, CAL_term.detach(), loss_rec_all.detach()
        else:
            return base_loss - self._alpha * peer_term, base_loss, peer_term

# ======================================================================== #
# Alpha Schedulers                                                         #
# ======================================================================== #
        
import math
from bisect import bisect_right

__all__ =[
    "StepAlpha", "MultiStepAlpha", "CosAnnealingAlpha", "SegAlpha"
]

class AlphaScheduler(object):
    def __init__(self, lossfunc, last_epoch=-1):
        '''
        control the variation of alpha during training
        NOTE, supposed to be used as follow:
            (othe code)
            for i in range(max_epoch):
                Train_epoch()
                Validate()
                AlphaScheduler.step()
        '''
        if ('peer loss function' not in lossfunc._name) or (not hasattr(lossfunc, "_name")):
            raise ValueError("AlphaScheduler only apply to Peer Loss")
        self._lossfunc = lossfunc
        self._base_alpha = self._lossfunc._alpha
        self._step_count = last_epoch + 1

    def get_alpha(self):
        raise NotImplementedError
    
    def step(self):
        self._step_count += 1
        self._lossfunc._alpha = self.get_alpha()


class StepAlpha(AlphaScheduler):
    '''
    Sets the alpha of Peer Lossf unction to the initial alpha
    decayed by gamma every step_size epochs.
    Args:
        lossfunc: Wrapped loss function.
        step_size (int): Period of alpha decay.
        gamma (float): Multiplicative factor of alpha decay.
            Default: 0.1.
    '''
    def __init__(self, lossfunc, step_size, gamma=0.1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepAlpha, self).__init__(lossfunc)

    def get_alpha(self):
        return self._base_alpha * \
               self.gamma ** (self._step_count // self.step_size)


class MultiStepAlpha(AlphaScheduler):
    '''
    Set the alpha of Peer Loss function to the initial alpha decayed
    by gamma once the number of epoch reaches one of the milestones.
    Args:
        lossfunc: Wrapped loss function.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of Alpha decay.
            Default: 0.1.
    '''
    def __init__(self, lossfunc, milestones, gamma=0.1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepAlpha, self).__init__(lossfunc)

    def get_alpha(self):
        return self._base_alpha * \
               self.gamma ** bisect_right(self.milestones, self._step_count)


class CosAnnealingAlpha(AlphaScheduler):
    '''
    Set the alpha of Peer Loss function using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial alpha and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    Args:
        lossfunc: Wrapped loss function.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum alpha value. Default: 0.
    '''
    def __init__(self, lossfunc, T_max, eta_min=0.):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosAnnealingAlpha, self).__init__(lossfunc)
    
    def get_alpha(self):
        return self.eta_min + (self._base_alpha - self.eta_min) * \
               (1 + math.cos(math.pi * self._step_count / self.T_max)) / 2


class SegAlpha(AlphaScheduler):
    '''
    Args:
        lossfunc: Wrapped loss function.
        alpha_list (list): different alphas.
        milestones (list): List of epoch indices. Must be increasing.
    '''
    def __init__(self, lossfunc, alpha_list, milestones, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        if not len(alpha_list) == len(milestones):
            raise ValueError('len(alpha_list) must be equal to len(milestones)')
        super(SegAlpha, self).__init__(lossfunc, last_epoch)
        self.alpha_list = alpha_list
        self.milestones = milestones
        self.len = len(alpha_list)
        self.slope = []
        self.alpha_cur = self._base_alpha
        alpha_cur = self._base_alpha
        epoch_cur = 0
        for i in range(self.len):
            slope = (alpha_list[i] - alpha_cur) / float(milestones[i] - epoch_cur)
            self.slope.append(slope)
            alpha_cur = alpha_list[i]
            epoch_cur = milestones[i]

    def get_alpha(self):
        idx = bisect_right(self.milestones, self._step_count - 1)
        if idx < self.len:
            self.alpha_cur += self.slope[idx]
        return self.alpha_cur