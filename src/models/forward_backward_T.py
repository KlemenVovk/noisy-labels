from typing import Any, Literal

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L

import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.models.resnet import resnet34

from utils.forward_backward_T import ForwardT, BackwardT, estimate_noise_mtx
from utils.noisylabels_resnet import ResNet34

# 2 optimizers -> one for warmup, one for loss corrected training
# that's why manual optimization
# on end of warmup, the loss switches from CE to corrected Loss
# this could be handled inside of the loss but it's not as transparent

# TODO: figure out how to set optimizer to another model

class ForwardBackwardT(L.LightningModule):

    def __init__(self, loss_type: Literal["forward", "backward"], n_warmup_epochs, initial_lr, momentum, weight_decay, datamodule):
        super().__init__()
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters(ignore=["loss_type", "datamodule"])
        self.automatic_optimization = False # manual forward due to multiple optimizers https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html

        self.num_training_samples = datamodule.num_training_samples
        self.num_classes = datamodule.num_classes
        #self.model = resnet34(weights=None, num_classes=self.num_classes) # don't use pretrained weights
        self.model = resnet34(weights=None, num_classes=self.num_classes)
        self.model_reinit = resnet34(weights=None, num_classes=self.num_classes)
        
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        
        # criterion changes after warmup
        self.n_warmup_epochs = n_warmup_epochs
        self.criterion = F.cross_entropy
        self.alt_criterion_cls = ForwardT if loss_type == "forward" else BackwardT
        self.training_step_outputs = [] # for noise estimation

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        # could be done with optim = self.optimizers()[self.current_epoch <= self.n_warmup_epochs]
        # for less code, but it's less readable imo
        if self.current_epoch <= self.n_warmup_epochs:
            opt, _ = self.optimizers()
        else:
            _, opt = self.optimizers()
        opt.zero_grad()

        # forward pass
        x, y_noise, y_true = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_noise)

        # backward pass
        loss.backward()
        opt.step()

        # save predictions on last warmup epoch
        if self.current_epoch == self.n_warmup_epochs:
            self.training_step_outputs.append(y_pred)

        # logging
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred, y_noise), on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        # step the scheduler
        # note that scheduler after warmup starts counting from 0
        if self.current_epoch <= self.n_warmup_epochs:
            sch, _ = self.lr_schedulers()
        else:
            _, sch = self.lr_schedulers()
        sch.step()

        # switch to noise-corrected loss
        if self.current_epoch == self.n_warmup_epochs:
            # estimate noise transition matrix
            all_preds = torch.cat(self.training_step_outputs, dim=0)
            X_prob = F.softmax(all_preds, dim=-1)
            T = estimate_noise_mtx(X_prob, filter_outlier=False)

            # switch criterion to corrected version
            self.criterion = self.alt_criterion_cls(T).to(self.device)

            # reinit the model
            self.model = self.model_reinit

            # clear training step outputs
            self.training_step_outputs = []
    
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_true = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(y_pred, y_true))
    
    def configure_optimizers(self):
        optim_args = dict(
            #params=self.model.parameters(),
            lr=self.hparams.initial_lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        optim_warmup = SGD(params=self.model.parameters(), **optim_args)
        optim = SGD(params=self.model_reinit.parameters(), **optim_args)

        # TODO: make pass schedule args at mudule init
        schedule_args = dict(
            milestones=[60],
            gamma=0.1
        )
        scheduler_warmup = MultiStepLR(optim_warmup, **schedule_args)
        scheduler = MultiStepLR(optim, **schedule_args)
        
        return [optim_warmup, optim], [scheduler_warmup, scheduler]
