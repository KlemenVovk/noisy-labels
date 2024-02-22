from typing import Any

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torchmetrics
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from methods.learning_strategies.base import LearningStrategyModule
from methods.learning_strategies.ELR.utils import elr_loss


class ELR(LearningStrategyModule):
    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args,
            optimizer_cls, optimizer_args, 
            scheduler_cls, scheduler_args, *args)
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        # initial_lr, momentum, weight_decay, beta, lmbd
        self.save_hyperparameters(ignore=["classifier_cls", "classifier_args", "datamodule", 
                                          "optimizer_cls", "optimizer_args", 
                                          "scheduler_cls", "scheduler_args"])
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes

        # init model
        self.model = classifier_cls(**classifier_args)
        self.criterion = elr_loss(self.num_training_samples, num_classes=self.num_classes, lmbd=self.hparams.lmbd, beta=self.hparams.beta)
        self.val_criterion = cross_entropy

        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
    

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noise, index = batch[0]
        y_pred = self.model(x)
        loss = self.criterion(index, y_pred, y_noise)
        # logging
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred, y_noise), on_epoch=True, on_step=False)
        return loss

    
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_true = batch
        y_pred = self.model(x)
        loss = self.val_criterion(y_pred, y_true)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(y_pred, y_true))
    

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_args)
        return [optimizer], [scheduler]