import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# losses


def loss_coteaching(y_1, y_2, t, forget_rate, ind):
    loss_1 = F.cross_entropy(y_1, t, reduction="none")
    ind_1_sorted = torch.argsort(loss_1)

    loss_2 = F.cross_entropy(y_2, t, reduction="none")
    ind_2_sorted = torch.argsort(loss_2)

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(ind_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return (
        torch.sum(loss_1_update) / num_remember,
        torch.sum(loss_2_update) / num_remember,
    )


def loss_coteaching_plus(logits, logits2, labels, forget_rate, ind, step):
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id = np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1):
        if p1 != pred2[idx]:
            disagree_id.append(idx)
            logical_disagree_id[idx] = True

    temp_disagree = ind.cpu().numpy() * logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0] == len(disagree_id)
    except AssertionError:
        disagree_id = disagree_id[: ind_disagree.shape[0]]

    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id]
        update_outputs2 = outputs2[disagree_id]

        loss_1, loss_2 = loss_coteaching(
            update_outputs, update_outputs2, update_labels, forget_rate, ind_disagree
        )
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2

        cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
        cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

        loss_1 = torch.sum(update_step * cross_entropy_1) / labels.size()[0]
        loss_2 = torch.sum(update_step * cross_entropy_2) / labels.size()[0]

    return loss_1, loss_2


# lr plan


def alpha_schedule(epoch, n_epochs=200, decay_start_epoch=80):
    if epoch >= decay_start_epoch:
        return float(n_epochs - epoch) / (n_epochs - decay_start_epoch)
    return 1
