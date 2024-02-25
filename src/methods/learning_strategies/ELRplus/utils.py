import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = torch.clip(torch.Tensor([current]), min=0.0, max=rampup_length)
        phase = 1.0 - current / rampup_length
        return float(torch.exp(-5.0 * phase * phase))
    

class elr_plus_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3, lmbd=3.0, coef_step=0, device='cuda'):
        super(elr_plus_loss, self).__init__()
        self.pred_hist = (torch.zeros(num_examp, num_classes)).to(device)
        self.q = 0
        self.beta = beta
        self.num_classes = num_classes
        self.lmbd = lmbd
        self.coef_step = coef_step


    def forward(self, iteration, output, y_labeled):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)

        if self.num_classes == 100:
            y_labeled = y_labeled * self.q
            y_labeled = y_labeled / (y_labeled).sum(dim=1,keepdim=True)

        ce_loss = torch.mean(-torch.sum(y_labeled * F.log_softmax(output, dim=1), dim = -1))
        reg = ((1-(self.q * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + sigmoid_rampup(iteration, self.coef_step)*(self.lmbd*reg)
      
        return  final_loss


    def update_hist(self, out, index= None, mix_index = ..., mixup_l = 1):
        y_pred_ = F.softmax(out,dim=1)
        self.pred_hist[index] = self.beta * self.pred_hist[index] +  (1-self.beta) *  y_pred_/(y_pred_).sum(dim=1,keepdim=True)
        self.q = mixup_l * self.pred_hist[index]  + (1-mixup_l) * self.pred_hist[index][mix_index]
    