from typing import Any, Type

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import torchmetrics
from torch.nn.functional import cross_entropy

from ..base import LearningStrategyModule


class CE(LearningStrategyModule):

    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: Type[Optimizer], optimizer_args: dict,
                 scheduler_cls: Type[LRScheduler], scheduler_args: dict,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args, 
            optimizer_cls, optimizer_args, scheduler_cls, scheduler_args, *args, **kwargs)
        
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        
        self.model = classifier_cls(**classifier_args)
        self.criterion = cross_entropy # basic CE
        
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch[0]
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred, y), on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(y_pred, y))

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_pred = self.model(x)
        self.test_acc(y_pred, y)
        self.log("test_acc", self.test_acc, on_epoch=True)
    
    def configure_optimizers(self):
        #optim = SGD(self.model.parameters(), lr=self.hparams.initial_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        #scheduler = MultiStepLR(optim, [60], 0.1)
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_args)
        return [optimizer], [scheduler]