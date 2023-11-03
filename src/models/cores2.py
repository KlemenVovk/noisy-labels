from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.models.resnet import resnet34
import lightning as L
import torch
import torchmetrics
from utils.cores2 import loss_cores, f_beta

# TODO: LR scheduling should be more flexible (current model works only for exactly 100 epochs like the authors proposed).
# TODO: implement cores2* - the second phase from the paper (consistency training).

# First phase from the paper: https://arxiv.org/abs/2010.02347
# Uses resnet34 as the backbone (not pretrained). Trained with CORES loss.
# Basically works on priors of the label noise and iteratively updates the priors after each epoch.
class SampleSieve(L.LightningModule):
    def __init__(self, initial_lr, momentum, weight_decay, num_classes, initial_noise_prior, num_training_samples):
        super().__init__()

        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters(ignore=["num_classes", "initial_noise_prior", "num_training_samples"])
        self.initial_noise_prior = initial_noise_prior
        self.cur_noise_prior = initial_noise_prior
        self.num_training_samples = num_training_samples
        self.num_classes = num_classes
        self.model = resnet34(weights=None, num_classes=num_classes) # don't use pretrained weights
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.noisy_class_frequency = torch.tensor([0] * self.num_classes).cuda()

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)        
        # clean_indicators is a list of 0s and 1s, where 1 means that the label is "predicted" to be clean, 0 means that the label is "predicted" to be noisy
        loss, clean_indicators = loss_cores(self.current_epoch, logits, y, noise_prior=self.cur_noise_prior)
        self.train_acc(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Record the frequency of predicted noisy classes
        for i, clean_indicator in enumerate(clean_indicators):
            if clean_indicator == 0:
                self.noisy_class_frequency[y[i]] += 1
        return loss

    def on_train_epoch_end(self):
        # Once the training epoch is done, update our prior of the label noise.
        self.cur_noise_prior = self.initial_noise_prior*self.num_training_samples - self.noisy_class_frequency
        self.cur_noise_prior = self.cur_noise_prior/sum(self.cur_noise_prior)
        self.cur_noise_prior = self.cur_noise_prior.cuda()
        self.noisy_class_frequency = torch.tensor([0] * self.num_classes).cuda()

    
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)
        loss, _ = loss_cores(self.current_epoch, logits, y)
        self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # Here multiple optimizers and schedulers can be set. Currently we have hardcoded the lr scheduling to exactly like it is in the paper.
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.initial_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        lr_plan = [0.1] * 50 + [0.01] * (50 + 1) # +1 because lr is set before the check if the training should stop due to reaching max_epochs (so its updated at the end of each epoch, for the next epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_plan[epoch]/(1+f_beta(epoch)))
        return [optimizer], [scheduler]