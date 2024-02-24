import torch 
import torch.nn.functional as F

def vanilla_loss(logits, labels):
    logprobs = F.log_softmax(y, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    loss = nll_loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    return torch.sum(loss)/num_batch


def loss_gls(logits, labels, smooth_rate=-0.2):
    confidence = 1. - smooth_rate
    logprobs = F.log_softmax(logits, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smooth_rate * smooth_loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    return torch.sum(loss)/num_batch
