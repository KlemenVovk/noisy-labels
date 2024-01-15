from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torchmetrics
from methods.learning_strategies.negative_ls.loss import vanilla_loss, nls_loss
from methods.learning_strategies.base import LearningStrategyModule

class NegativeLS(LearningStrategyModule):
    
    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args,
            optimizer_cls, optimizer_args, 
            scheduler_cls, scheduler_args)
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters()

        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        self.criterion = vanilla_loss
        
        # init model
        self.model = classifier_cls(**classifier_args)
        
        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.noisy_class_frequency = torch.zeros(self.num_classes)
    
    def on_train_epoch_end(self):
        if self.current_epoch == self.hparams["warmup_epochs"]:
            self.criterion = nls_loss
            # reset optimizer and scheduler
            # TODO: NegativeLS doesn't work yet, I have to find a way to set up 2 different schedulers with 2 different lr plans...nicely.
            self.configure_optimizers()

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        [[x, y]] = batch
        logits = self.model(x)        
        # clean_indicators is a list of 0s and 1s, where 1 means that the label is "predicted" to be clean, 0 means that the label is "predicted" to be noisy
        loss = self.criterion(logits, y)
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
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_args)
        return [optimizer], [scheduler]