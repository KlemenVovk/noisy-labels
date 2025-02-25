import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Taken from: https://github.com/haochenglouis/cores

def loss_cores(epoch, y, t, noise_prior = None):
    beta = f_beta(epoch)
    loss = F.cross_entropy(y, t, reduction="none")
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0)) # ???
    loss_ = -torch.log(F.softmax(y, dim=-1) + 1e-8)
    # sel metric
    loss_sel =  loss - torch.mean(loss_,1) 
    if noise_prior is None:
        loss =  loss - beta*torch.mean(loss_,1)
    else:
        loss =  loss - beta*torch.sum(torch.mul(noise_prior, loss_),1)
    
    loss_div_numpy = loss_sel.data.cpu().numpy()
    for i in range(len(loss_numpy)):
        if epoch <= 30: # NOTE: first 10 epochs are warmup and we use only cross entropy loss
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000
    else:
        return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)

def f_beta(epoch):
    beta1 = np.linspace(0.0, 0.0, num=10)
    beta2 = np.linspace(0.0, 2, num=30)
    beta3 = np.linspace(2, 2, num=60 + 1) # +1 because lr is set before the check if the training should stop due to reaching max_epochs (so its updated at the end of each epoch, for the next epoch)
 
    beta = np.concatenate((beta1,beta2,beta3),axis=0) 
    return beta[epoch]

