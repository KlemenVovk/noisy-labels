import torch
import torch.nn.functional as F


BATCH_MAP = {
    "warmup1": 0,
    "warmup2": 1,
    "labeled1": 0,
    "labeled2": 1,
    "unlabeled1": 2,
    "unlabeled2": 3,
    "eval_train": 4,
}


def end_warmup(datamodule):
    datamodule.train_datasets[BATCH_MAP["warmup1"]].mode = 'labeled'
    datamodule.train_datasets[BATCH_MAP["warmup2"]].mode = 'labeled'

        
def set_probabilities(datamodule, prob1, prob2):
    datamodule.train_datasets[BATCH_MAP["labeled1"]].probabilities = prob1
    datamodule.train_datasets[BATCH_MAP["labeled2"]].probabilities = prob2
    datamodule.train_datasets[BATCH_MAP["unlabeled1"]].probabilities = prob1
    datamodule.train_datasets[BATCH_MAP["unlabeled2"]].probabilities = prob2


def set_predictions(datamodule, pred1, pred2):
    datamodule.train_datasets[BATCH_MAP["labeled1"]].prediction = pred1
    datamodule.train_datasets[BATCH_MAP["labeled2"]].prediction = pred2
    labeled_indices1 = torch.arange(len(pred1))[pred1]
    unlabeled_indices1 = torch.arange(len(pred1))[~pred1]
    labeled_indices2 = torch.arange(len(pred2))[pred2]
    unlabeled_indices2 = torch.arange(len(pred2))[~pred2]
    datamodule.train_datasets[BATCH_MAP["labeled1"]].labeled_indices = labeled_indices1
    datamodule.train_datasets[BATCH_MAP["labeled1"]].unlabeled_indices = unlabeled_indices1
    datamodule.train_datasets[BATCH_MAP["labeled2"]].labeled_indices = labeled_indices2
    datamodule.train_datasets[BATCH_MAP["labeled2"]].unlabeled_indices = unlabeled_indices2
    # set unlabeled dataset indices
    datamodule.train_datasets[BATCH_MAP["unlabeled1"]].unlabeled_indices = unlabeled_indices1
    datamodule.train_datasets[BATCH_MAP["unlabeled2"]].unlabeled_indices = unlabeled_indices2


def linear_rampup(current, warm_up, rampup_length=16, lambda_u=25):
    current = torch.clip(torch.tensor(current-warm_up) / rampup_length, min=0.0, max=1.0)
    return lambda_u*float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)
    

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
    