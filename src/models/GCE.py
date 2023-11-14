from typing import Any

import os
from pathlib import Path

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L

from torchvision.models.resnet import resnet34
import torch
import torchmetrics
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from utils.GCE import GCELoss


class GCE(L.LightningModule):
    def __init__(self, prune_start_epoch, prune_freq, checkpoint_dir, initial_lr, momentum, weight_decay, datamodule):
        super().__init__()
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters(ignore=["checkpoin_dir", "datamodule"])
        self.num_training_samples = datamodule.num_training_samples
        self.num_classes = datamodule.num_classes
        self.datamodule = datamodule
        self.model = resnet34(weights=None, num_classes=self.num_classes) # don't use pretrained weights
        #self.model = ResNet34(num_classes=self.num_classes) # noisylabels resnet performs much better

        self.criterion = GCELoss() # generalized CE
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")

        # directory for saving best model
        self.best_acc = 0
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_model_path = self.checkpoint_dir / "best_model.pt"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_train_epoch_start(self) -> None:
        epoch = self.current_epoch
        prune_start = self.hparams.prune_start_epoch
        freq = self.hparams.prune_freq
        if epoch >= prune_start and (epoch-prune_start) % freq == 0:
            print("pruning... loss")
            best_model = torch.load(self.best_model_path)
            best_model.to(self.device)
            best_model.eval()

            # run train loader on best model and update weights
            for batch in self.datamodule.train_dataloader(): # TODO: possible problem with multiple dataloaders in future
                batch = [t.to(self.device) for t in batch]
                x, y_noise, y_true, idxs = batch
                y_pred = best_model(x)
                self.criterion.update_weight(y_pred, y_noise, idxs)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noise, _, idxs = batch
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
        optim = SGD(self.model.parameters(), lr=self.hparams.initial_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = MultiStepLR(optim, [60], 0.1)
        return [optim], [scheduler]
