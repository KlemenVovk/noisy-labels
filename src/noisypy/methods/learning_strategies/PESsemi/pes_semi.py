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
from torch.nn.functional import cross_entropy, log_softmax, one_hot, softmax

from ..base import LearningStrategyModule
from .utils import update_train_data_and_criterion, update_dataloaders, linear_rampup


class PES_semi(LearningStrategyModule):

    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type[Module], classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 PES_lr: float, T1: int, T2: int, lambda_u: float, 
                 temperature: float, alpha: float,
                 optimizer_refine_cls: type[Optimizer],
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args, 
            optimizer_cls, optimizer_args, scheduler_cls, scheduler_args, *args, **kwargs)
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        # PES_lr, T1, T2, lambda_u, alpha, temperature
        self.save_hyperparameters(ignore=["classifier_cls", "classifier_args", "datamodule", 
                                          "optimizer_cls", "optimizer_args", 
                                          "scheduler_cls", "scheduler_args", "optimizer_refine_cls"])
        
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        self.num_iter = datamodule.num_train_samples // datamodule.batch_size
        self.PES_lr = PES_lr
        self.T1 = T1
        self.T2 = T2
        self.lambda_u = lambda_u
        self.temperature = temperature
        self.alpha = alpha
        self.optimizer_refine_cls = optimizer_refine_cls
        
        # init model
        self.model = classifier_cls(**classifier_args)
        self.criterion = cross_entropy 
        self.val_criterion = cross_entropy
        
        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")

    def on_train_epoch_start(self) -> None:
        # save training data, noisy labels and train transform for noisy refinement and dataset updating
        if self.current_epoch == 0:
            train_dataset = self.trainer.datamodule.train_datasets[0]
            self.train_data = train_dataset.data
            self.train_labels = []
            for i in tqdm(range(len(train_dataset)), desc=f'Saving Training Data', leave=False):
                _, y, *_ = train_dataset[i]
                self.train_labels.append(y)
            self.train_labels = torch.LongTensor(self.train_labels)
            self.train_transform = train_dataset.transform
            
        if self.current_epoch == self.T1:
            self.model = self.noisy_refine(self.model, num_layer=0, refine_times=self.T2)

    def noisy_refine(self, model: Module, num_layer: int, refine_times: int) -> Module:
        if refine_times <= 0:
            return model
        
        # freeze all layers and add a new final layer
        for param in model.parameters():
            param.requires_grad = False

        model.renew_layers(num_layer)
        device = next(self.model.parameters()).device
        model.to(device)
        
        optimizer_refine = self.optimizer_refine_cls(model.parameters(), lr=self.PES_lr)
        train_loader = self.original_trainloader
        for epoch in range(refine_times):
            for batch in tqdm(train_loader, desc=f'Noisy Refine {epoch+1}/{refine_times}', leave=False):
                loss = self.train_step(move_data_to_device(batch, device))
                optimizer_refine.zero_grad()
                loss.backward()
                optimizer_refine.step()

        del self.original_trainloader

        # unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True

        return model

    def train_step(self, batch: Any) -> STEP_OUTPUT:
        x, y_noise = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_noise) 

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_pred, y_noise), on_epoch=True, on_step=False)
        return loss   

    def on_train_epoch_end(self) -> None:
        # prepare data for the next epoch for every epoch after (including) T1
        if self.current_epoch + 1 == self.T1:
            self.original_trainloader = self.datamodule.train_dataloader()[0]
        if self.current_epoch + 1 >= self.T1:
            # update train dataset and criterion
            labeled_trainloader, unlabeled_trainloader, class_weights = update_train_data_and_criterion(
                self.model, train_data=self.train_data, noisy_targets=self.train_labels, 
                transform_train=self.train_transform, batch_size=self.datamodule.batch_size)
        
            update_dataloaders(self.datamodule, labeled_trainloader, unlabeled_trainloader)
            self.class_weights = class_weights

    def mix_match_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        inputs_x, inputs_x2, targets_x = batch[0]
        inputs_u, inputs_u2 = batch[1]

        batch_size = inputs_x.size(0)
        targets_x = one_hot(targets_x, num_classes=self.num_classes)

        with torch.no_grad():
            outputs_u11 = self.model(inputs_u)
            outputs_u12 = self.model(inputs_u2)

            pu = (softmax(outputs_u11, dim=1) + softmax(outputs_u12, dim=1)) / 2
            ptu = pu**(1 / self.temperature)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixmatch_l = torch.distributions.beta.Beta(self.alpha, self.alpha).sample().item()
        mixmatch_l = max(mixmatch_l, 1 - mixmatch_l)

        mixed_input = mixmatch_l * input_a + (1 - mixmatch_l) * input_b
        mixed_target = mixmatch_l * target_a + (1 - mixmatch_l) * target_b

        logits = self.model(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx_mean = -torch.mean(log_softmax(logits_x, dim=1) * mixed_target[:batch_size * 2], 0)
        Lx = torch.sum(Lx_mean * self.class_weights)

        probs_u = torch.softmax(logits_u, dim=1)
        Lu = torch.mean((probs_u - mixed_target[batch_size * 2:])**2)
        loss = Lx + linear_rampup(self.current_epoch + batch_idx / self.num_iter, self.T1, lambda_u=self.lambda_u) * Lu

        self.log("mix_match_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        if self.current_epoch < self.T1:
            return self.train_step(batch[0])
        else:
            return self.mix_match_step(batch, batch_idx)

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
        self.log("test_acc", self.test_acc(y_pred, y))

    def configure_optimizers(self) -> list[list[Optimizer], list[LRScheduler]]:
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_args)
        return [optimizer], [scheduler]