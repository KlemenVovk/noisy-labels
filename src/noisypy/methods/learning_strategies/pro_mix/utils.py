import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningDataModule


BATCH_MAP = {
    "warmup": 0,
    "train": 0,
    "eval_train": 1,
}


class ProMixModel(nn.Module):
    def __init__(
        self, base_model_cls, model_type="paper_resnet", feat_dim=128, **kwargs
    ):
        super(ProMixModel, self).__init__()
        self.model = base_model_cls(**kwargs)
        self.base_model_cls = base_model_cls

        self.model_type = model_type
        if model_type == "paper_resnet":
            dim_in = self.model.linear.in_features
            num_classes = self.model.linear.out_features
        elif model_type == "pytorch_resnet":
            dim_in = self.model.fc.in_features
            num_classes = self.model.fc.out_features
        else:
            raise ValueError(f"model_type {model_type} not supported")

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        self.pseudo_linear = nn.Linear(dim_in, num_classes)

    def forward(self, x, train=False, use_ph=False):
        if self.model_type == "paper_resnet":
            out = F.relu(self.model.bn1(self.model.conv1(x)))
            out = self.model.layer1(out)
            out = self.model.layer2(out)
            out = self.model.layer3(out)
            out = self.model.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out_linear = self.model.linear(out)
        elif self.model_type == "pytorch_resnet":
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            out = torch.flatten(x, 1)
            out_linear = self.model.fc(out)
        else:
            raise ValueError(f"model_type {self.model_type} not supported")

        if train:
            feat_c = self.head(out)
            if use_ph:
                out_linear_debias = self.pseudo_linear(out)
                return out_linear, out_linear_debias, F.normalize(feat_c, dim=1)
            else:
                return out_linear, F.normalize(feat_c, dim=1)
        else:
            if use_ph:
                out_linear_debias = self.pseudo_linear(out)
                return out_linear, out_linear_debias
            else:
                return out_linear


class NegEntropy(object):
    def __call__(self, outputs):
        outputs = outputs.clamp(min=1e-12)
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


class CE_Soft_Label(nn.Module):
    def __init__(self):
        super().__init__()
        self.confidence = None

    def init_confidence(self, noisy_labels, num_class):
        noisy_labels = torch.Tensor(noisy_labels).long().cuda()
        self.confidence = F.one_hot(noisy_labels, num_class).float().clone().detach()

    def forward(self, outputs, targets=None):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * targets.detach()
        loss_vec = -((final_outputs).sum(dim=1))
        return loss_vec

    @torch.no_grad()
    def confidence_update(self, temp_un_conf, batch_index, conf_ema_m):
        with torch.no_grad():
            _, prot_pred = temp_un_conf.max(dim=1)
            pseudo_label = (
                F.one_hot(prot_pred, temp_un_conf.shape[1]).float().cuda().detach()
            )
            self.confidence[batch_index, :] = (
                conf_ema_m * self.confidence[batch_index, :]
                + (1 - conf_ema_m) * pseudo_label
            )
        return None


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def debias_pl(logit, bias, tau=0.4):
    bias = bias.detach().clone()
    debiased_prob = F.softmax(logit - tau * torch.log(bias), dim=1)
    return debiased_prob


def debias_output(logit, bias, tau=0.8):
    bias = bias.detach().clone()
    debiased_opt = logit + tau * torch.log(bias)
    return debiased_opt


def bias_initial(num_class=10):
    bias = (torch.ones(num_class, dtype=torch.float) / num_class).cuda()
    return bias


def bias_update(input, bias, momentum, bias_mask=None):
    if bias_mask is not None:
        input_mean = input.detach() * bias_mask.detach().unsqueeze(dim=-1)
    else:
        input_mean = input.detach().mean(dim=0)
    bias = momentum * bias + (1 - momentum) * input_mean
    return bias


# Data processing
def end_warmup(datamodule: LightningDataModule):
    datamodule.train_datasets[BATCH_MAP["train"]].mode = "all_lab"
    datamodule.train_datasets[BATCH_MAP["eval_train"]].mode = "all"
    datamodule.train_datasets[
        BATCH_MAP["eval_train"]
    ].transform = datamodule.test_datasets[0].transform


def co_divide(
    datamodule: LightningDataModule, prob1: torch.Tensor, prob2: torch.Tensor
):
    datamodule.train_datasets[BATCH_MAP["train"]].probabilities = prob1
    datamodule.train_datasets[BATCH_MAP["train"]].probabilities2 = prob2
