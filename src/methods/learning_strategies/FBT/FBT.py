from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L

import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .utils import ForwardTLoss, BackwardTLoss, estimate_noise_mtx
from ..base import LearningStrategyWithWarmupModule


class ForwardT(LearningStrategyWithWarmupModule):

    def __init__(self, 
                 datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 warmup_epochs: int, 
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args,
            optimizer_cls, optimizer_args, scheduler_cls,
            scheduler_args, warmup_epochs, *args, **kwargs)
        
        # so we can mess around with schedulers and optimizers
        self.automatic_optimization = False

        # standard stuff
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        self.model = classifier_cls(**classifier_args)
        self.model_reinit = classifier_cls(**classifier_args)
        
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        
        # criterion changes after warmup
        self.warmup_epochs = warmup_epochs - 1 if warmup_epochs > 1 else 0 # :)
        self.criterion = F.cross_entropy
        self.alt_criterion_cls = ForwardTLoss
        self.training_step_outputs = [] # for noise estimation

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        # could be done with optim = self.optimizers()[self.current_epoch <= self.n_warmup_epochs]
        # for less code, but it's less readable imo
        if self.current_epoch <= self.warmup_epochs:
            opt, _ = self.optimizers()
        else:
            _, opt = self.optimizers()
        opt.zero_grad()

        # forward pass
        x, y_noise = batch[0]
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_noise)

        # backward pass
        self.manual_backward(loss)
        opt.step()

        # save predictions on last warmup epoch
        if self.current_epoch == self.warmup_epochs:
            self.training_step_outputs.append(y_pred)

        # logging
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred, y_noise), on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        # step the scheduler
        # note that scheduler after warmup starts counting from 0
        if self.current_epoch <= self.warmup_epochs:
            sch, _ = self.lr_schedulers()
        else:
            _, sch = self.lr_schedulers()
        sch.step()

        # switch to noise-corrected loss
        if self.current_epoch == self.warmup_epochs:
            print("switching to noise corrected training...")

            # estimate noise transition matrix
            all_preds = torch.cat(self.training_step_outputs, dim=0)
            X_prob = F.softmax(all_preds, dim=-1)
            T = estimate_noise_mtx(X_prob, filter_outlier=False)

            # switch criterion to corrected version
            self.criterion = self.alt_criterion_cls(T).to(self.device)

            # reinit the model
            self.model = self.model_reinit

            # clear training step outputs and old model
            self.training_step_outputs = []
    
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_true = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(y_pred, y_true))
    
    def configure_optimizers(self):
        optim_warmup = self.optimizer_cls(params=self.model.parameters(), **self.optimizer_args)
        optim = self.optimizer_cls(params=self.model_reinit.parameters(), **self.optimizer_args)

        scheduler_warmup = self.scheduler_cls(optim_warmup, **self.scheduler_args)
        scheduler = self.scheduler_cls(optim, **self.scheduler_args)
        
        return [optim_warmup, optim], [scheduler_warmup, scheduler]    


class BackwardT(ForwardT):

    def __init__(self, 
                 datamodule: L.LightningDataModule, 
                 classifier_cls: type, classifier_args: dict, 
                 optimizer_cls: type[Optimizer], optimizer_args: dict, 
                 scheduler_cls: type[LRScheduler], scheduler_args: dict, 
                 warmup_epochs: int, 
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args, 
            optimizer_cls, optimizer_args, scheduler_cls, 
            scheduler_args, warmup_epochs, *args, **kwargs)
        self.alt_criterion_cls = BackwardTLoss
