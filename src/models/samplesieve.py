from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.models.resnet import resnet34
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchmetrics
from utils.cores_loss import loss_cores, f_beta

# TODO: aim logger

class SampleSieve(L.LightningModule):
    def __init__(self, initial_lr, momentum, weight_decay, num_classes):
        super().__init__()
        self.save_hyperparameters() # saves arguments passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.model = resnet34(pretrained=True)
        # Replace the last layer with a new one with num_classes outputs
        self.model.fc = nn.Linear(self.model.fc.in_features, self.hparams.num_classes)
        self.train_precision_at_1 = torchmetrics.Precision(num_classes=self.hparams.num_classes, top_k=1, task='multiclass')
        self.val_precision_at_1 = torchmetrics.Precision(num_classes=self.hparams.num_classes, top_k=1, task='multiclass')

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)
        loss = loss_cores(self.current_epoch, logits, y)
        self.log('train_loss', loss)
        self.train_precision_at_1(logits, y)
        self.log('train_precision_at_1', self.train_precision_at_1, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)
        loss = loss_cores(self.current_epoch, logits, y)
        self.log('val_loss', loss)
        self.val_precision_at_1(logits, y)
        self.log('val_precision_at_1', self.val_precision_at_1, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        # SGD
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.initial_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        lr_plan = [0.1] * 50 + [0.01] * 50 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_plan[epoch]/(1+f_beta(epoch)))
        return [optimizer], [scheduler]