from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torchmetrics
from methods.learning_strategies.negative_ls.loss import vanilla_loss, loss_gls
from methods.learning_strategies.base import LearningStrategyWithWarmupModule

class NegativeLS(LearningStrategyWithWarmupModule):
    
    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: list[type[Optimizer]],
                 optimizer_args: list[dict],
                 scheduler_cls: list[type[LRScheduler]],
                 scheduler_args: list[dict],
                 warmup_epochs: int,
                 smooth_rate: float,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args,
            optimizer_cls,
            optimizer_args,
            scheduler_cls,
            scheduler_args,
            warmup_epochs, *args, **kwargs)

        self.automatic_optimization = False

        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters()

        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        self.criterion = lambda logits, y: vanilla_loss(logits, y)
        self.stage = 0
        
        # init model
        self.model = classifier_cls(**classifier_args)
        
        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
    
    def on_train_epoch_end(self):
        if self.current_epoch == self.hparams["warmup_epochs"] - 1: # switch to the next optimizer and scheduler
            self.criterion = lambda logits, y: loss_gls(logits, y, self.hparams.smooth_rate)
            self.stage = 1
        else:
            # step the scheduler
            scheduler = self.lr_schedulers()[self.stage]
            scheduler.step()


    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        [[x, y]] = batch
        optimizer = self.optimizers()[self.stage]
        optimizer.zero_grad()
        logits = self.model(x)        
        # clean_indicators is a list of 0s and 1s, where 1 means that the label is "predicted" to be clean, 0 means that the label is "predicted" to be noisy
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.manual_backward(loss)
        optimizer.step()
        return loss
    

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizers = [self.optimizer_cls(self.model.parameters(), **optimizer_args) for optimizer_args in self.optimizer_args]
        schedulers = [self.scheduler_cls(optimizer, **scheduler_args) for scheduler_args, optimizer in zip(self.scheduler_args, optimizers)]
        return optimizers, schedulers