from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L
import torch
import torchmetrics
from methods.learning_strategies.cores2.utils import loss_cores, f_beta

# TODO: LR scheduling should be more flexible (current model works only for exactly 100 epochs like the authors proposed).
# TODO: implement cores2* - the second phase from the paper (consistency training).
# TODO: log the warmup!
# TODO: maybe it would be good to log the accuracy with the noisy labels as a sanity check to see if the loss is working and making a difference (because it can happen that a NN is just powerful enough to work even with a bad/baseline loss)

# First phase from the paper: https://arxiv.org/abs/2010.02347
# Uses resnet34 as the backbone (not pretrained). Trained with CORES loss.
# Basically works on priors of the label noise and iteratively updates the priors after each epoch.

class SampleSieve(L.LightningModule):
    def __init__(self, 
                 classifier_cls, classifier_args, 
                 datamodule,
                 optimizer_cls, optimizer_args,
                 scheduler_cls, scheduler_args):
        super().__init__()
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        # self.save_hyperparameters(ignore=["classifier_cls", "classifier_args", "datamodule"])

        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        self._compute_initial_noise_prior(datamodule)
        
        # init model
        self.model = classifier_cls(**classifier_args)

        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_args = scheduler_args
        
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.noisy_class_frequency = torch.zeros(self.num_classes)
        
    def _compute_initial_noise_prior(self, datamodule):
        # Noise prior is just the class probabilities
        train_dataset = datamodule.train_datasets[0]
        class_frequency = torch.zeros(self.num_classes)
        for i in range(len(train_dataset)):
            y = train_dataset[i][1]
            class_frequency[y] += 1
        self.initial_noise_prior = class_frequency / class_frequency.sum()
        self.initial_noise_prior = self.initial_noise_prior
        self.cur_noise_prior = self.initial_noise_prior

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        [[x, y]] = batch
        logits = self.model(x)        
        # clean_indicators is a list of 0s and 1s, where 1 means that the label is "predicted" to be clean, 0 means that the label is "predicted" to be noisy
        loss, clean_indicators = loss_cores(self.current_epoch, logits, y, noise_prior=self.cur_noise_prior.to(self.device))
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
        self.cur_noise_prior = self.initial_noise_prior * self.num_training_samples - self.noisy_class_frequency
        self.cur_noise_prior = self.cur_noise_prior / sum(self.cur_noise_prior)
        self.cur_noise_prior = self.cur_noise_prior
        self.noisy_class_frequency = torch.zeros(self.num_classes)

    
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)
        loss, _ = loss_cores(self.current_epoch, logits, y, noise_prior=self.cur_noise_prior.to(self.device))
        self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # Here multiple optimizers and schedulers can be set. Currently we have hardcoded the lr scheduling to exactly like it is in the paper.
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.initial_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        #lr_plan = [0.1] * 50 + [0.01] * (50 + 1) # +1 because lr is set before the check if the training should stop due to reaching max_epochs (so its updated at the end of each epoch, for the next epoch)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_plan[epoch]/(1+f_beta(epoch)))
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_args)
        return [optimizer], [scheduler]