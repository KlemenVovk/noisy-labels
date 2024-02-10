from typing import Any

import lightning as L
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import torchmetrics
from methods.learning_strategies.base import LearningStrategyModule

from .utils import loss_coteaching

# NOTE: needs AddIndex dataset augmentation

class CoTeaching(LearningStrategyModule):

    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 forget_rate: float, exponent: float, num_gradual: int,
                 num_epochs: int, # TODO: figure out how to remove num of epochs
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args, 
            optimizer_cls, optimizer_args, scheduler_cls, scheduler_args, *args, **kwargs)
        
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        
        self.model1 = classifier_cls(**classifier_args)
        self.model2 = classifier_cls(**classifier_args)
        self.criterion = loss_coteaching # basic CE
        
        # metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")

        # forget rate scheduling
        self.rate_schedule = torch.ones(num_epochs) * forget_rate
        self.rate_schedule[:num_gradual] = torch.linspace(0, forget_rate ** exponent, num_gradual)

        # turn off auto optim so we can cook
        self.automatic_optimization = False

    def training_step(self, batch: Any, batch_idx: int) -> None:
        optim1, optim2 = self.optimizers()
        x, y_noise, idxs = batch[0]
        
        # forward
        logits1 = self.model1(x)
        logits2 = self.model2(x)

        # calculate losses
        loss1, loss2 = self.criterion(
            logits1, logits2, y_noise, 
            self.rate_schedule[self.current_epoch], idxs)
        
        # backward and step
        optim1.zero_grad()
        loss1.backward()
        optim1.step()
        
        optim2.zero_grad()
        loss2.backward()
        optim2.step()
        
        # log allat
        self.log("train_loss1", loss1, prog_bar=True)
        self.log("train_loss2", loss2, prog_bar=True)
        self.log("train_acc1", self.train_acc(logits1, y_noise), on_epoch=True, on_step=False)
        self.log("train_acc2", self.train_acc(logits2, y_noise), on_epoch=True, on_step=False)
    
    def on_train_epoch_end(self) -> None:
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y_true = batch
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        self.log("val_acc1", self.val_acc(logits1, y_true))
        self.log("val_acc2", self.val_acc(logits2, y_true))
    
    def configure_optimizers(self):
        optim1 = self.optimizer_cls(self.model1.parameters(), **self.optimizer_args)
        optim2 = self.optimizer_cls(self.model2.parameters(), **self.optimizer_args)

        scheduler1 = self.scheduler_cls(optim1, **self.scheduler_args)
        scheduler2 = self.scheduler_cls(optim2, **self.scheduler_args)        
        return [optim1, optim2], [scheduler1, scheduler2]