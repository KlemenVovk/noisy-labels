from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L

import torchmetrics
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR


class CE(L.LightningModule):
    def __init__(self, 
                 classifier_cls, classifier_args,
                 datamodule,
                 optimizer_cls, optimizer_args,
                 scheduler_cls, scheduler_args):
        super().__init__()
        
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        
        self.model = classifier_cls(**classifier_args)
        self.criterion = cross_entropy # basic CE
        
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_args = scheduler_args
        
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noise = batch[0]
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_noise)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred, y_noise), on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_true = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(y_pred, y_true))
    
    def configure_optimizers(self):
        #optim = SGD(self.model.parameters(), lr=self.hparams.initial_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        #scheduler = MultiStepLR(optim, [60], 0.1)
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_args)
        return [optimizer], [scheduler]