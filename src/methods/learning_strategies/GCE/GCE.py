from typing import Any
import os
from pathlib import Path

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L

import torch
import torchmetrics
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


from .utils import GCELoss
from ..base import LearningStrategyModule


class GCE(LearningStrategyModule):

    def __init__(self, 
                 datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 checkpoint_dir: str,
                 prune_start_epoch: int, prune_freq: int,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args,
            optimizer_cls, optimizer_args, scheduler_cls,
            scheduler_args, *args, **kwargs)

        self.save_hyperparameters("prune_start_epoch", "prune_freq")

        # model
        self.num_classes = self.datamodule.num_classes
        self.model = self.classifier_cls(**self.classifier_args)

        # loss and metrics
        self.criterion = GCELoss(trainset_size=self.datamodule.num_train_samples)
        self.train_acc = torchmetrics.Accuracy(
            num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(
            num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")

        # setup for saving best model
        self.best_acc = 0
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_model_path = self.checkpoint_dir / "best_model.pt"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_train_epoch_start(self) -> None:
        # prune loss
        epoch = self.current_epoch
        prune_start = self.hparams.prune_start_epoch
        freq = self.hparams.prune_freq
        if epoch >= prune_start and (epoch-prune_start) % freq == 0:
            print("pruning...")

            # load best model
            best_model = torch.load(self.best_model_path)
            best_model.to(self.device)
            best_model.eval()

            # run train loader on best model and update weights
            for batch in self.datamodule.train_dataloader()[0]:
                batch = [t.to(self.device) for t in batch]
                x, y_noise, idxs = batch
                y_pred = best_model(x)
                self.criterion.update_weight(y_pred, y_noise, idxs)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noise, idxs = batch[0]
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_noise, idxs)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred, y_noise), on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_true = batch
        y_pred = self.model(x)
        # there are different idxs for train and val
        # so it makes no sense to calculate val GCE loss
        #loss = self.criterion(y_pred, y_true) 
        #self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(y_pred, y_true))

    def on_validation_end(self) -> None:
        # save model if it has highest val_acc
        val_acc = self.trainer.callback_metrics["val_acc"] # maybe there is a cleaner way
        if val_acc > self.best_acc:
            torch.save(self.model, self.best_model_path)
            self.best_acc = val_acc
    
    def configure_optimizers(self):
        optim = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_cls(optim, **self.scheduler_args)
        return [optim], [scheduler]
