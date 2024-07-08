from typing import Any

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torchmetrics
import torch
from torch.nn.functional import cross_entropy, one_hot
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..base import LearningStrategyModule
from .utils import overparametrization_loss


class SOP(LearningStrategyModule):
    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 ratio_consistency: float, ratio_balance: float, lr_u: float, lr_v: float,
                 overparam_optimizer_cls: type[Optimizer], overparam_weight_decay: float, 
                 overparam_momentum: float, overparam_mean: float, overparam_std: float,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args,
            optimizer_cls, optimizer_args, 
            scheduler_cls, scheduler_args, *args, **kwargs)
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        # ratio_consistency, ratio_balance, lr_u, lr_v, overparam_weight_decay, overparam_momentum, overparam_mean, overparam_std
        self.save_hyperparameters(ignore=["classifier_cls", "classifier_args", "datamodule", 
                                          "optimizer_cls", "optimizer_args", 
                                          "scheduler_cls", "scheduler_args", "overparam_optimizer_cls"])
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes

        # init model
        self.ratio_consistency = ratio_consistency
        self.lr_u = lr_u
        self.lr_v = lr_v
        self.overparam_momentum = overparam_momentum
        self.overparam_weight_decay = overparam_weight_decay
        self.overparam_optimizer_cls = overparam_optimizer_cls
        self.model = classifier_cls(**classifier_args)
        self.criterion = overparametrization_loss(num_examp=self.num_training_samples, num_classes=self.num_classes, 
                                                  ratio_consistency=self.ratio_consistency, ratio_balance=ratio_balance,
                                                  mean=overparam_mean, std=overparam_std)
        self.val_criterion = cross_entropy

        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        
        self.automatic_optimization = False

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noise, index = batch[0]
        opt, opt_loss = self.optimizers()
        self.model.train()

        y = one_hot(y_noise, self.num_classes).float()

        y_pred = self.model(x)
        loss = self.criterion(index, y_pred, y)
        self.train_acc(y_pred, y_noise)

        # optimization step
        opt_loss.zero_grad()
        opt.zero_grad()
        loss.backward()
        opt_loss.step()
        opt.step()
        # logging
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self) -> None:
        scheduler = self.lr_schedulers()
        scheduler.step()

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_true = batch
        y_pred = self.model(x)
        loss = self.val_criterion(y_pred, y_true)
        self.val_acc(y_pred, y_true)
        self.log("val_loss", loss)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_pred = self.model(x)
        self.test_acc(y_pred, y)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def configure_optimizers(self) -> list[list[Optimizer], list[LRScheduler]]:
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_args)
        overparametrization_params = [
            {'params': self.criterion.u, 'lr': self.lr_u, 'weight_decay': self.overparam_weight_decay, 'momentum': self.overparam_momentum},
            {'params': self.criterion.v, 'lr': self.lr_v, 'weight_decay': self.overparam_weight_decay, 'momentum': self.overparam_momentum}
        ]
        optimizer_loss = self.overparam_optimizer_cls(overparametrization_params)
        return [optimizer, optimizer_loss], [scheduler]