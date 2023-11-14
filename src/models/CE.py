from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L

from torchvision.models.resnet import resnet34
import torch
import torchmetrics
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR


class CE(L.LightningModule):
    def __init__(self, initial_lr, momentum, weight_decay, datamodule):
        super().__init__()
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters(ignore=["datamodule"])
        self.num_training_samples = datamodule.num_training_samples
        self.num_classes = datamodule.num_classes
        self.model = resnet34(weights=None, num_classes=self.num_classes) # don't use pretrained weights
        #self.model = ResNet34(num_classes=self.num_classes) # noisylabels resnet performs much better
        self.criterion = cross_entropy # basic CE
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noise, _ = batch
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
        optim = SGD(self.model.parameters(), lr=self.hparams.initial_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = MultiStepLR(optim, [60], 0.1)
        return [optim], [scheduler]
