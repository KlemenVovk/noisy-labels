from typing import Any

import lightning as L
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torchmetrics

from ..base import LearningStrategyModule
from .utils import loss_jocor


class JoCoR(LearningStrategyModule):

    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 forget_rate: float, exponent: float, num_gradual: int,
                 co_lambda: float,
                 num_epochs: int,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args,
            optimizer_cls, optimizer_args, scheduler_cls, scheduler_args, *args, **kwargs)
        self.save_hyperparameters("forget_rate", "exponent", "num_gradual", "co_lambda")
               
        self.model1 = classifier_cls(**classifier_args)
        self.model2 = classifier_cls(**classifier_args)
        self.criterion = loss_jocor
        self.co_lambda = co_lambda
        
        # metrics
        N = datamodule.num_classes
        self.train_acc = torchmetrics.Accuracy(
            num_classes=N, top_k=1, task='multiclass', average="micro"
        )
        self.val_acc = torchmetrics.Accuracy(
            num_classes=N, top_k=1, task='multiclass', average="micro"
        )

        # forget rate scheduling
        self.forget_rate_schedule = torch.ones(num_epochs) * forget_rate
        self.forget_rate_schedule[:num_gradual] = torch.linspace(
            0, forget_rate ** exponent, num_gradual
        )

    def training_step(self, batch: Any, batch_idx: int) -> None:
        x, y_noise = batch[0]
        
        # forward
        logits1 = self.model1(x)
        logits2 = self.model2(x)

        # calculate losses
        loss = self.criterion(
            logits1, logits2, y_noise, 
            self.forget_rate_schedule[self.current_epoch],
            self.co_lambda
        )
               
        # log allat
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc1", self.train_acc(logits1, y_noise), on_epoch=True, on_step=False)
        self.log("train_acc2", self.train_acc(logits2, y_noise), on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y_true = batch
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        self.log("val_acc1", self.val_acc(logits1, y_true))
        self.log("val_acc2", self.val_acc(logits2, y_true))
    
    def configure_optimizers(self):
        optim = self.optimizer_cls(
            list(self.model1.parameters()) + list(self.model2.parameters()), 
            **self.optimizer_args
        )
        scheduler = self.scheduler_cls(optim, **self.scheduler_args)
        return [optim], [scheduler]
