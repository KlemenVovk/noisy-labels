import torch
import torch.nn as nn
import torch.nn.functional as F


class overparametrization_loss(nn.Module):
    def __init__(
        self,
        num_examp,
        num_classes=10,
        ratio_consistency=0,
        ratio_balance=0,
        mean=0.0,
        std=1e-8,
        eps=1e-4,
    ):
        super(overparametrization_loss, self).__init__()
        self.num_classes = num_classes
        self.num_examp = num_examp

        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance
        self.mean = mean
        self.std = std
        self.eps = eps

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))

        self.init_param(mean=self.mean, std=self.std)

    def init_param(self, mean=0.0, std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, outputs, label):
        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
        else:
            output = outputs

        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)

        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)

        E = U_square - V_square
        self.E = E

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(
            original_prediction + U_square - V_square.detach(), min=self.eps
        )
        prediction = F.normalize(prediction, p=1, eps=self.eps)
        prediction = torch.clamp(prediction, min=self.eps, max=1.0)
        label_one_hot = self.soft_to_hard(output.detach())
        MSE_loss = F.mse_loss(
            (label_one_hot + U_square - V_square), label, reduction="sum"
        ) / len(label)
        loss = torch.mean(-torch.sum((label) * torch.log(prediction), dim=-1))
        loss += MSE_loss

        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min=self.eps, max=1.0)
            balance_kl = torch.mean(
                -(prior_distr * torch.log(avg_prediction)).sum(dim=0)
            )
            loss += self.ratio_balance * balance_kl

        if (len(outputs) > len(index)) and (self.ratio_consistency > 0):
            consistency_loss = self.consistency_loss(output, output2)
            loss += self.ratio_consistency * torch.mean(consistency_loss)

        return loss

    def consistency_loss(self, output1, output2):
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction="none")
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv

    def soft_to_hard(self, x):
        with torch.no_grad():
            return F.one_hot(x.argmax(dim=1), self.num_classes).float()


class LabelParameterization(nn.Module):
    def __init__(self, n_samples, n_class, init="gaussian", mean=0.0, std=1e-4):
        super(LabelParameterization, self).__init__()
        self.n_samples = n_samples
        self.n_class = n_class
        self.init = init

        self.s = nn.Parameter(torch.empty(n_samples, n_class, dtype=torch.float32))
        self.t = nn.Parameter(torch.empty(n_samples, n_class, dtype=torch.float32))
        self.history = torch.zeros(n_samples, n_class, dtype=torch.float32).cuda()
        self.init_param(mean=mean, std=std)

    def init_param(self, mean=0.0, std=1e-4):
        if self.init == "gaussian":
            torch.nn.init.normal_(self.s, mean=mean, std=std)
            torch.nn.init.normal_(self.s, mean=mean, std=std)
        elif self.init == "zero":
            torch.nn.init.constant_(self.s, 0)
            torch.nn.init.constant_(self.t, 0)
        else:
            raise TypeError("Label not initialized.")

    def compute_loss(self):
        param_y = self.s * self.s - self.t * self.t
        return torch.linalg.norm(param_y, ord=2) ** 2

    def forward(self, feature, idx):
        y = feature

        param_y = self.s[idx] * self.s[idx] - self.t[idx] * self.t[idx]

        history = 0.3 * param_y + 0.7 * self.history[idx]

        self.history[idx] = history.detach()

        assert param_y.shape == y.shape, "Label and param shape do not match."
        return y + history, y


def reparameterization(
    n_samples=50000, num_classes=10, init="gaussian", mean=0.0, std=1e-4
):
    return LabelParameterization(
        n_samples=n_samples, n_class=num_classes, init=init, mean=mean, std=std
    )
