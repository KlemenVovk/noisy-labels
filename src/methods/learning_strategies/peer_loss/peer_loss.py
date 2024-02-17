from typing import Any
from copy import deepcopy
from functools import partial

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torchmetrics

from methods.learning_strategies.base import MultiStageLearningStrategyModule
from .utils import CrossEntropyLossStable, f_alpha


class PeerLoss(MultiStageLearningStrategyModule):

    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: list[type[Optimizer]], optimizer_args: list[dict],
                 scheduler_cls: list[type[LRScheduler]], scheduler_args: list[dict],
                 stage_epochs: list[int],
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args, 
            optimizer_cls, optimizer_args, scheduler_cls, scheduler_args,
            stage_epochs,
            *args, **kwargs)

        # model
        self.model = classifier_cls(**classifier_args)

        # criterions
        self.criterion = CrossEntropyLossStable()
        
        # metrics
        N = self.datamodule.num_classes
        self.train_acc = torchmetrics.Accuracy(num_classes=N, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=N, top_k=1, task='multiclass', average="micro")

        # misc
        self.peer_factor_plan = partial(f_alpha, r=0.1)
        self._best_model = None
        self._best_val_acc = 0
       
    def configure_optimizers(self):
        # optimizers
        optim_stage0 = self.optimizer_cls[0](
            self.model.parameters(),
            **self.optimizer_args[0]
        )
        optim_stage1 = self.optimizer_cls[1](
            self.model.parameters(),
            **self.optimizer_args[1]
        )

        # schedulers
        sch_stage0 = self.scheduler_cls[0](optim_stage0, **self.scheduler_args[0])
        sch_stage1 = self.scheduler_cls[1](optim_stage1, **self.scheduler_args[1])
        return [optim_stage0, optim_stage1], [sch_stage0, sch_stage1]
    
    ###################
    # Stage 0 - Warmup
    ###################
    
    def training_step_stage0(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noise = batch[0]
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_noise)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred, y_noise), on_epoch=True, on_step=False)
        return loss
    
    def validation_step_stage0(self, batch: Any, batch_idx: int) -> None:
        x, y_noise = batch
        y_pred = self.model(x)
        self.log("val_acc", self.val_acc(y_pred, y_noise))
    
    def on_validation_end_stage0(self) -> None:
        # save model if it has the highest val_acc
        val_acc = self.trainer.callback_metrics["val_acc"]
        if val_acc > self._best_val_acc:
            self._best_model = deepcopy(self.model)
            self._best_val_acc = val_acc
        
        # at the end, of stage switch the model to the best one
        # and reset best model and best val accuracy
        if self.current_epoch == self.stage_epoch_cumsum[0]:
            self.model.load_state_dict(self._best_model.state_dict())
            self._best_model = None
            self._best_val_acc = 0

    #####################
    # Stage 1 - Training
    #####################
    
    def training_step_stage1(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noise = batch[0]
        x_sh, y_noise_sh = batch[1]

        y_pred = self.model(x)
        y_pred_sh = self.model(x_sh)
        
        fac = self.peer_factor_plan(self.current_epoch)
        loss = self.criterion(y_pred, y_noise) -\
            fac * self.criterion(y_pred_sh, y_noise_sh)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred, y_noise), on_epoch=True, on_step=False)
        return loss
    
    def on_train_epoch_end_stage0(self) -> None:
        self.log("f_alpha", self.peer_factor_plan(self.current_epoch))
        self.log("lr", self.optimizers()[1].param_groups[0]["lr"])
    
    def validation_step_stage1(self, batch: Any, batch_idx: int) -> None:
        x, y_noise = batch
        y_pred = self.model(x)
        self.log("val_acc", self.val_acc(y_pred, y_noise))
    
    def on_validation_end_stage1(self) -> None:
        # save model if it has the highest val_acc
        val_acc = self.trainer.callback_metrics["val_acc"]
        if val_acc > self._best_val_acc:
            self._best_model = deepcopy(self.model)
            self._best_val_acc = val_acc
