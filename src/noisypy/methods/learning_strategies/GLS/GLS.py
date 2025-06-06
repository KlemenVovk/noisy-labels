from copy import deepcopy
from typing import Any, Type
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torchmetrics
from .utils import loss_gls
from ..base import LearningStrategyModule
import torch.nn.functional as F

class GLS(LearningStrategyModule):
    
    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: list[Type[Optimizer]] | Type[Optimizer],
                 optimizer_args: list[dict] | dict,
                 scheduler_cls: list[Type[LRScheduler]] | Type[LRScheduler],
                 scheduler_args: list[dict] | dict,
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

        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters(ignore=["classifier_cls", "classifier_args", "datamodule", 
                                          "optimizer_cls", "optimizer_args", 
                                          "scheduler_cls", "scheduler_args"])

        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes

        self.criterion = F.cross_entropy
        self.stage = 0
        # The authors also run some models directly with GLS loss, without warmup.
        if warmup_epochs == -1:
            self.criterion = lambda logits, y: loss_gls(logits, y, self.hparams.smooth_rate)
            self.stage = 1
        
        # init model
        self.model = classifier_cls(**classifier_args)
        self._best_model = None
        self._best_val_acc = 0
        
        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
    
    def on_train_epoch_end(self):
        if self.current_epoch == self.hparams["warmup_epochs"] - 1: # switch to the next optimizer and scheduler
            print("Switching to GLS loss.")
            self.stage = 1
            self.criterion = lambda logits, y: loss_gls(logits, y, self.hparams.smooth_rate)
            # load best model from warmup
            self.model.load_state_dict(self._best_model.state_dict())
        
        if self.current_epoch != self.trainer.max_epochs - 1 and self.current_epoch != self.hparams["warmup_epochs"] - 1: 
            scheduler = self.lr_schedulers()[self.stage] if self.hparams["warmup_epochs"] != -1 else self.lr_schedulers()
            scheduler.step()


    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        [[x, y]] = batch
        logits = self.model(x)        
        loss = self.criterion(logits, y)
        optimizer = self.optimizers()[self.stage] if self.hparams["warmup_epochs"] != -1 else self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        # loss.backward()
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
    
    def on_validation_epoch_end(self) -> None:
        val_acc = self.trainer.callback_metrics["val_acc"]
        if val_acc > self._best_val_acc and self.stage == 0:
            self._best_model = deepcopy(self.model)
            self._best_val_acc = val_acc
    
    def configure_optimizers(self):
        # No warmup
        if self.hparams["warmup_epochs"] == -1:
            optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_args)
            return [optimizer], [scheduler]
        
        # Warmup + main
        optimizer_warmup = self.optimizer_cls(self.model.parameters(), **self.optimizer_args[0])
        optimizer_main = self.optimizer_cls(self.model.parameters(), **self.optimizer_args[1])
        scheduler_warmup = self.scheduler_cls(optimizer_warmup, **self.scheduler_args[0])
        scheduler_main = self.scheduler_cls(optimizer_main, **self.scheduler_args[1])
        return [optimizer_warmup, optimizer_main], [scheduler_warmup, scheduler_main]