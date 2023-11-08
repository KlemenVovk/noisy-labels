from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.models.resnet import resnet34
import lightning as L
import torch
import torchmetrics
from utils.fdivergence import CrossEntropyLossStable, ProbLossStable, Divergence


class FDivergence(L.LightningModule):
    def __init__(self, initial_lr, momentum, weight_decay, datamodule, divergence, warmup_epochs):
        super().__init__()
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters(ignore=["datamodule"])
        self.num_training_samples = datamodule.num_training_samples
        self.num_classes = datamodule.num_classes
        # TODO: enable setting an arbitrary backbone model
        self.model = resnet34(weights=None, num_classes=self.num_classes)  # don't use pretrained weights
        # metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task="multiclass")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task="multiclass")
        
        # TODO: should specific properties of the model be set here or somewhere else?
        self.divergence = Divergence(divergence)
        self.criterion = CrossEntropyLossStable()
        self.criterion_prob = ProbLossStable()
        self.warmup_epochs = warmup_epochs
        self.warmup = False

    def on_train_epoch_start(self):
        if self.current_epoch < self.warmup_epochs:
            self.warmup = True
        else:
            self.warmup = False
    
    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:        
        (x_1, y_1_noisy, y_1_true), (x_2, y_2_noisy, y_2_true) = batch
        logits_1 = self.model(x_1)
        
        # TODO: problems
        # 1) we need 2 dataloaders for noisy train set
        # 2) we compute loss differently for warmup
        if self.warmup:
            # regular CE with noisy labels (NOTE: their implementation of CE)
            loss = self.criterion(logits_1, y_1_noisy)
        else:
            prob_reg = - self.criterion_prob(logits_1, y_1_noisy)
            loss_regular = self.divergence.activation(prob_reg)
            
            logits_2 = self.model(x_2)
            prob_peer = - self.criterion_prob(logits_2, y_2_noisy)
            loss_peer = self.divergence.conjugate(prob_peer)
            
            loss = loss_regular - loss_peer
        
        self.train_acc(logits_1, y_1_true)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)
        
        # TODO: which loss should we report here? The authors only compute accuracy in the validation and test steps...
        # just report the regular CE loss for now
        loss = self.criterion(logits, y)

        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Here multiple optimizers and schedulers can be set. Currently we have hardcoded the lr scheduling to exactly like it is in the paper.
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.initial_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        lr_plan = [1e-2] * 30 + [1e-3] * 30 + [1e-4] * 30 + [1e-5] * 30 + [1e-6] * (30 + 1)  # +1 because lr is set before the check if the training should stop due to reaching max_epochs (so its updated at the end of each epoch, for the next epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_plan[epoch])
        return [optimizer], [scheduler]
