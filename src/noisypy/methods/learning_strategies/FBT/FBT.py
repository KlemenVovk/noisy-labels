from typing import Any, Type
from copy import deepcopy

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
    def __init__(
        self,
        datamodule: L.LightningDataModule,
        classifier_cls: type,
        classifier_args: dict,
        optimizer_cls: Type[Optimizer],
        optimizer_args: dict,
        scheduler_cls: Type[LRScheduler],
        scheduler_args: dict,
        warmup_epochs: int,
        filter_outliers: bool,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            datamodule,
            classifier_cls,
            classifier_args,
            optimizer_cls,
            optimizer_args,
            scheduler_cls,
            scheduler_args,
            warmup_epochs,
            *args,
            **kwargs,
        )
        self.save_hyperparameters("warmup_epochs", "filter_outliers")

        # so we can mess around with schedulers and optimizers
        self.automatic_optimization = False

        # standard stuff
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        self.model = classifier_cls(**classifier_args)

        self.train_acc = torchmetrics.Accuracy(
            num_classes=self.num_classes, top_k=1, task="multiclass"
        )
        self.val_acc = torchmetrics.Accuracy(
            num_classes=self.num_classes, top_k=1, task="multiclass"
        )
        self.test_acc = torchmetrics.Accuracy(
            num_classes=self.num_classes, top_k=1, task="multiclass", average="micro"
        )

        # criterion changes after warmup
        self.warmup_epochs = warmup_epochs - 1 if warmup_epochs > 1 else 0  # :)
        self.filter_outliers = filter_outliers
        self.criterion = F.cross_entropy
        self.alt_criterion_cls = ForwardTLoss

        # estimating T
        self._best_model = None
        self._best_val_acc = 0
        self._probs = []
        self.T = None

    @property
    def stage(self):
        return self.current_epoch <= self.warmup_epochs

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        opt = self.optimizers()[self.stage]
        opt.zero_grad()

        # forward pass
        x, y_noise = batch[0]
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_noise)

        # backward pass
        self.manual_backward(loss)
        opt.step()

        # save predictions
        self._probs.append(F.softmax(y_pred.detach(), dim=-1))

        # logging
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_acc", self.train_acc(y_pred, y_noise), on_epoch=True, on_step=False
        )
        return loss

    def on_train_epoch_end(self):
        # step the scheduler
        # note that scheduler after warmup starts counting from 0
        sch = self.lr_schedulers()[self.stage]
        sch.step()

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_true = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(y_pred, y_true))

    def on_validation_end(self) -> None:
        # save model if it has highest val_acc
        val_acc = self.trainer.callback_metrics["val_acc"]
        if val_acc > self._best_val_acc:
            self._best_model = deepcopy(self.model)
            self._best_val_acc = val_acc
            self.T = self._estimate_T()

        if self.current_epoch == self.warmup_epochs:
            print("Switching to noise corrected training.")
            # reinit the model and clear buffers
            self.model.load_state_dict(
                self.classifier_cls(**self.classifier_args).state_dict()
            )
            self._best_model = None
            # switch criterion to corrected version
            self.criterion = self.alt_criterion_cls(self.T).to(self.device)
        self._probs = []

    def _estimate_T(self):
        if self._probs:
            probs = torch.cat(self._probs, dim=0)
            return estimate_noise_mtx(probs).to(self.device)
        return torch.eye(self.num_classes).to(self.device)

    def configure_optimizers(self):
        optim_warmup = self.optimizer_cls(
            params=self.model.parameters(), **self.optimizer_args
        )
        optim = self.optimizer_cls(
            params=self.model.parameters(), **self.optimizer_args
        )

        scheduler_warmup = self.scheduler_cls(optim_warmup, **self.scheduler_args)
        scheduler = self.scheduler_cls(optim, **self.scheduler_args)

        return [optim_warmup, optim], [scheduler_warmup, scheduler]


class BackwardT(ForwardT):
    def __init__(
        self,
        datamodule: L.LightningDataModule,
        classifier_cls: type,
        classifier_args: dict,
        optimizer_cls: Type[Optimizer],
        optimizer_args: dict,
        scheduler_cls: Type[LRScheduler],
        scheduler_args: dict,
        warmup_epochs: int,
        filter_outliers: bool,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            datamodule,
            classifier_cls,
            classifier_args,
            optimizer_cls,
            optimizer_args,
            scheduler_cls,
            scheduler_args,
            warmup_epochs,
            filter_outliers,
            *args,
            **kwargs,
        )
        self.alt_criterion_cls = BackwardTLoss
