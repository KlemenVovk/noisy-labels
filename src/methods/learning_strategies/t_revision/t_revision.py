from typing import Any
from itertools import accumulate
from copy import deepcopy

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer, Adam, SGD
from torch.optim.lr_scheduler import LRScheduler
import torchmetrics
from torch.nn.functional import cross_entropy

from methods.learning_strategies.base import LearningStrategyModule
from .utils import ReweightLoss, ReweightRevisionLoss
from ..FBT.utils import estimate_noise_mtx

# TODO: fix the optimizers so they are not hardcoded

class TRevision(LearningStrategyModule):

    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 stage_epochs: list[int],
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args, 
            optimizer_cls, optimizer_args, scheduler_cls, scheduler_args, *args, **kwargs)
        
        N = self.datamodule.num_classes
        self.num_classes = N
        self.stage_epoch_cumsum = list(accumulate(stage_epochs))

        # model
        self.model = classifier_cls(**classifier_args)

        # criterions
        self.criterion_stage0 = cross_entropy
        self.criterion_stage1 = ReweightLoss()
        self.criterion_stage2 = ReweightRevisionLoss()
        
        # metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=N, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=N, top_k=1, task='multiclass', average="micro")

        # auto opt OFF
        self.automatic_optimization = False

        #TODO: maybe it's better to dump model to disk
        # misc
        self._best_model = None # buffer for best model
        self._best_val_acc = 0
        self._probs = []
        self.T = None
        self.T_revision = nn.Parameter(torch.zeros((N, N)))
    
    @property
    def current_stage(self):
        valid = [i for i, e in enumerate(self.stage_epoch_cumsum) if self.current_epoch <= e]
        return min(valid)
    
    def _get_current_stage_method(self, method_name):
        # returns the method: method_name_stage{current_stage}
        # or a sink for compatibility
        entries = [e for e in dir(self) if method_name+"_stage" in e]
        entries = sorted(entries, key=lambda e: int(e[-1]))
        if len(entries) > self.current_stage:
            method_name = entries[self.current_stage]
            return getattr(self, method_name)
        return lambda *_: None # just a dummy sink function

    def training_step(self, batch: Any, batch_idx: int) -> None:
        # get step and optim for current stage
        self.log("stage", self.current_stage)
        training_step = self._get_current_stage_method("training_step")
        optim = self.optimizers()[self.current_stage] 
        
        # training_step + optimize
        loss = training_step(batch, batch_idx)
        optim.zero_grad()
        loss.backward()
        optim.step()

    def on_train_epoch_end(self) -> None:
        on_train_epoch_end = self._get_current_stage_method("on_train_epoch_end")
        sch = self.lr_schedulers()[self.current_stage]
        
        on_train_epoch_end()
        sch.step()
    
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        validation_step = self._get_current_stage_method("validation_step")
        validation_step(batch, batch_idx)

    def on_validation_end(self) -> None:
        on_validation_end = self._get_current_stage_method("on_validation_end")
        on_validation_end()
    
    def configure_optimizers(self):
        # optimizers
        optim_stage0 = SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        optim_stage1 = SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
        optim_stage2 = Adam(
            list(self.model.parameters()) + [self.T_revision],
            lr=5e-7, weight_decay=1e-4,
        )

        # schedulers
        sch_stage0 = self.scheduler_cls(optim_stage0, **self.scheduler_args)
        sch_stage1 = self.scheduler_cls(optim_stage1, **self.scheduler_args)
        sch_stage2 = self.scheduler_cls(optim_stage2, **self.scheduler_args)
        return [optim_stage0, optim_stage1, optim_stage2], [sch_stage0, sch_stage1, sch_stage2]
    
    ###################
    # Stage 0 - Warmup
    ###################
    
    def training_step_stage0(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noise = batch[0]
        y_pred = self.model(x)
        loss = self.criterion_stage0(y_pred, y_noise)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred, y_noise), on_epoch=True, on_step=False)
        self._probs.append(F.softmax(y_pred.detach(), dim=-1))
        return loss
    
    def validation_step_stage0(self, batch: Any, batch_idx: int) -> None:
        x, y_noise = batch
        y_pred = self.model(x)
        loss = self.criterion_stage0(y_pred, y_noise)

        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(y_pred, y_noise))
    
    def on_validation_end_stage0(self) -> None:
        # save model if it has highest val_acc, estimate transition mtx T
        val_acc = self.trainer.callback_metrics["val_acc"]
        if val_acc > self._best_val_acc:
            self._best_model = deepcopy(self.model)
            self._best_val_acc = val_acc
            self.T = self._estimate_T()
        
        # at the end, of stage switch the model to the best one
        # and reset best model and best val accuracy
        if self.current_epoch == self.stage_epoch_cumsum[0]:
            self.model.load_state_dict(self._best_model.state_dict())
            self._best_model = None
            self._best_val_acc = 0
        self._probs = []
    
    def _estimate_T(self):
        if self._probs:
            probs = torch.cat(self._probs, dim=0)
            return estimate_noise_mtx(probs).to(self.device)
        return torch.eye(self.num_classes).to(self.device)

    #####################
    # Stage 1 - Training
    #####################
    
    def training_step_stage1(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noise = batch[0]
        y_pred = self.model(x)
        loss = self.criterion_stage1(y_pred, self.T, y_noise)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred @ self.T, y_noise), on_epoch=True, on_step=False)
        return loss
    
    def validation_step_stage1(self, batch: Any, batch_idx: int) -> None:
        x, y_noise = batch
        y_pred = self.model(x)
        loss = self.criterion_stage1(y_pred, self.T, y_noise)

        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(y_pred @ self.T, y_noise))
    
    def on_validation_end_stage1(self) -> None:
        # save model if it has highest val_acc
        val_acc = self.trainer.callback_metrics["val_acc"]
        if val_acc > self._best_val_acc:
            self._best_model = deepcopy(self.model)
            self._best_val_acc = val_acc
        
        # at the end, of stage switch the model to best one
        # and reset best model and best val accuracy
        if self.current_epoch == self.stage_epoch_cumsum[1]:
            self.model.load_state_dict(self._best_model.state_dict())
            self._best_model = None
            self._best_val_acc = 0

    #####################
    # Stage 2 - Revision
    #####################
    
    def training_step_stage2(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noise = batch[0]
        y_pred = self.model(x)
        loss = self.criterion_stage2(y_pred, self.T, self.T_revision, y_noise)

        T = self.T + self.T_revision
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred @ T, y_noise), on_epoch=True, on_step=False)
        return loss
    
    def validation_step_stage2(self, batch: Any, batch_idx: int) -> None:
        x, y_noise = batch
        y_pred = self.model(x)
        loss = self.criterion_stage2(y_pred, self.T, self.T_revision, y_noise)

        T = self.T + self.T_revision
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(y_pred @ T, y_noise))
