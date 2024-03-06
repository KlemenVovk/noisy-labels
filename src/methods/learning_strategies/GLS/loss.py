import torch 
import torch.nn.functional as F

def loss_gls(epoch, y, t, smooth_rate=0.1, wa=0, wb=1):
    confidence = 1. - smooth_rate
    logprobs = F.log_softmax(y, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=t.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = (wa + wb * confidence) * nll_loss + wb * smooth_rate * smooth_loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    return torch.sum(loss)/num_batch
 