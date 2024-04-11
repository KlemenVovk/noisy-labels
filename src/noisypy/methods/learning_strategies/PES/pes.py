from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities import move_data_to_device
import lightning as L
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import torchmetrics
from torch.nn.functional import cross_entropy

from ..base import LearningStrategyModule
from .utils import update_train_data_and_criterion, update_dataloader, renew_layers


class PES(LearningStrategyModule):

    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type[Module], classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 PES_lr: float, warmup_epochs: int, T2: int, T3: int,
                 optimizer_refine_cls: type[Optimizer],
                 model_type: str = 'pytorch_resnet',
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args, 
            optimizer_cls, optimizer_args, scheduler_cls, scheduler_args, *args, **kwargs)
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        # PES_lr, warmup_epochs, T2, T3
        self.save_hyperparameters(ignore=["classifier_cls", "classifier_args", "datamodule", 
                                          "optimizer_cls", "optimizer_args", 
                                          "scheduler_cls", "scheduler_args", "optimizer_refine_cls"])
        
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        self.PES_lr = PES_lr
        self.warmup_epochs = warmup_epochs
        self.T2 = T2
        self.T3 = T3
        self.optimizer_refine_cls = optimizer_refine_cls
        self.model_type = model_type
        
        # init model
        self.model = classifier_cls(**classifier_args)
        self.criterion = cross_entropy
        self.val_criterion = cross_entropy
        
        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")

        self.automatic_optimization = False

    def on_train_epoch_start(self) -> None:
        self.model.train()
        # save training data, noisy labels and train transform for noisy refinement and dataset updating
        if self.current_epoch == 0:
            train_dataset = self.trainer.datamodule.train_datasets[0]
            index = train_dataset.valid_idxs
            self.train_data = train_dataset.data[index]
            self.train_labels = []
            for i in tqdm(index, desc=f'Saving Training Data', leave=False):
                _, y, *_ = train_dataset[i]
                self.train_labels.append(y)
            self.train_labels = torch.LongTensor(self.train_labels)
            self.train_transform = train_dataset.transform
        if self.current_epoch == self.warmup_epochs:
            self.model = self.noisy_refine(self.model, num_layer=1, refine_times=self.T2)
            self.model = self.noisy_refine(self.model, num_layer=0, refine_times=self.T3)

    def on_train_epoch_end(self) -> None:
        if self.current_epoch >= self.warmup_epochs:
            # update train dataset and criterion
            # https://github.com/tmllab/2021_NeurIPS_PES/blob/ec5290d9fcc9efa8f302dbe8a78c448805d9e6e7/PES_cs.py#L179
            confident_dataset, train_criterion = update_train_data_and_criterion(self.model, train_data=self.train_data, 
                            noisy_targets=self.train_labels, transform_train=self.train_transform, 
                            batch_size=self.datamodule.batch_size)
            
            update_dataloader(self.datamodule, confident_dataset)
            self.criterion = train_criterion
        
        scheduler = self.lr_schedulers()
        scheduler.step()

    def noisy_refine(self, model: Module, num_layer: int, refine_times: int) -> Module:
        if refine_times <= 0:
            return model
        
        # freeze all layers and add a new final layer
        for param in model.parameters():
            param.requires_grad = False

        model = renew_layers(model, last_num_layers=num_layer, model_class=self.model_type)
        device = next(self.model.parameters()).device
        model.to(device)
        
        optimizer_refine = self.optimizer_refine_cls(model.parameters(), lr=self.PES_lr)

        train_loader = self.datamodule.train_dataloader()[0]
        for epoch in range(refine_times):
            model.train()
            for batch in tqdm(train_loader, desc=f'Noisy Refine {epoch+1}/{refine_times}', leave=False):
                self.train_step(move_data_to_device(batch, device), model, optimizer_refine)

        for param in model.parameters():
            param.requires_grad = True

        return model

    def train_step(self, batch: Any, model: Module, optimizer: Optimizer) -> STEP_OUTPUT:
        x, y_noise = batch
        y_pred = model(x)
        loss = self.criterion(y_pred, y_noise) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred, y_noise), on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        if self.current_epoch == self.warmup_epochs:
            # only noise refinement is done in this epoch, skip training step
            # https://github.com/tmllab/2021_NeurIPS_PES/blob/ec5290d9fcc9efa8f302dbe8a78c448805d9e6e7/PES_cs.py#L173-L180
            return
        return self.train_step(batch[0], self.model, self.optimizers())

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_true = batch
        y_pred = self.model(x)
        self.val_acc(y_pred, y_true)
        loss = self.val_criterion(y_pred, y_true)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_pred = self.model(x)
        self.test_acc(y_pred, y)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def configure_optimizers(self) -> list[list[Optimizer], list[LRScheduler]]:
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_args)
        return [optimizer], [scheduler]