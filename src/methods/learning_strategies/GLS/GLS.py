from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torchmetrics
from methods.learning_strategies.GLS.loss import vanilla_loss, loss_gls
from methods.learning_strategies.base import LearningStrategyModule
import torch.nn.functional as F

class GLS(LearningStrategyModule):
    
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
            *args, **kwargs)

        self.automatic_optimization = False
        self.stage = 0

        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters()

        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        self.criterion = F.cross_entropy
        
        # init model
        self.model = classifier_cls(**classifier_args)
        
        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
    
    def on_train_epoch_end(self):
        if self.current_epoch == self.hparams["warmup_epochs"] - 1: # switch to the next optimizer and scheduler
            print("Switching to GLS loss.")
            self.stage = 1
            self.criterion = lambda logits, y: loss_gls(0, logits, y, self.hparams.smooth_rate)
        
        if self.current_epoch != self.trainer.max_epochs - 1 and self.current_epoch != self.hparams["warmup_epochs"] - 1: 
            scheduler = self.lr_schedulers()[self.stage]
            scheduler.step()


    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        [[x, y]] = batch
        logits = self.model(x)        
        loss = self.criterion(logits, y)
        optimizer = self.optimizers()[self.stage]
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        self.train_acc(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
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
        optimizer_warmup = self.optimizer_cls(self.model.parameters(), **self.optimizer_args[0])
        optimizer_main = self.optimizer_cls(self.model.parameters(), **self.optimizer_args[1])
        scheduler_warmup = self.scheduler_cls(optimizer_warmup, **self.scheduler_args[0])
        scheduler_main = self.scheduler_cls(optimizer_main, **self.scheduler_args[1])
        return [optimizer_warmup, optimizer_main], [scheduler_warmup, scheduler_main]