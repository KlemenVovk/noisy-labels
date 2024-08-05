from typing import Any, Type

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L

from tqdm import tqdm
import torch.nn as nn
from torch.nn.functional import softmax
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torchmetrics

from ..base import LearningStrategyModule
from .utils import PeerLossRegCE

## CAL imports
import torch

EPS = 1e-8

class CAL(LearningStrategyModule):

    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 alpha: float, warmup_epochs: int,
                 alpha_scheduler_cls: type, alpha_scheduler_args: dict,
                 alpha_scheduler_args_warmup: dict,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args, 
            optimizer_cls, optimizer_args, scheduler_cls, scheduler_args, *args, **kwargs)
        
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes

        self.alpha = alpha
        self.warmup_epochs = warmup_epochs
        self.alpha_scheduler_cls = alpha_scheduler_cls
        self.alpha_scheduler_args = alpha_scheduler_args
        
        self.weight = torch.ones(self.num_training_samples)
        self.distilled_weight = torch.ones(self.num_training_samples)
        self.distilled_weight_new = torch.zeros(self.num_training_samples)
        self.distilled_label = torch.zeros(self.num_training_samples).long()
        self.distilled_label_new = torch.zeros(self.num_training_samples).long()
        self.train_labels = []
        for i in tqdm(range(len(datamodule.train_datasets[0])), desc=f'Saving Training Labels', leave=False):
            _, y, *_ = datamodule.train_datasets[0][i]
            self.train_labels.append(y)
        self.train_labels = torch.LongTensor(self.train_labels)
        # square root noise prior
        self.class_size_noisy = torch.bincount(self.train_labels)
        self.noisy_prior = self.class_size_noisy / self.num_training_samples
        self.noisy_prior = torch.sqrt(self.noisy_prior) / torch.sqrt(self.noisy_prior).sum()


        self.loss_mean_all = torch.zeros((self.num_classes, self.num_classes))

        self.model = classifier_cls(**classifier_args)
        self.model_warmup = classifier_cls(**classifier_args)

        self.criterion = PeerLossRegCE(alpha, self.noisy_prior, 'crossentropy')
        self.alpha_scheduler = alpha_scheduler_cls(self.criterion, **alpha_scheduler_args_warmup)
        self.val_criterion = nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")

        self.automatic_optimization = False


    def on_train_epoch_start(self) -> None:
        self.loss_rec_all = torch.zeros((self.num_classes, self.num_classes)).to(self._device) * 1.0
        self.loss_mean_all = self.loss_mean_all.to(self.device)
        self.criterion._peer._prior = self.criterion._peer._prior.to(self.device)
        self.distilled_weight_new = self.distilled_weight_new.to(self.device)
        self.distilled_label_new = self.distilled_label_new.to(self.device)
        if self.current_epoch >= self.warmup_epochs:
            self.T_mat_indicator_sum = torch.max(torch.sum(torch.sum(torch.sum(self.T_mat>0.0,2),1).view(self.T_mat.shape[0],1,-1).repeat(1,self.T_mat.shape[0],1),2) * 1.0, torch.ones((self.T_mat.shape[0],self.T_mat.shape[1])).to(self.T_mat.device))
            

    def on_train_epoch_end(self) -> None:
        if self.current_epoch + 1 == self.warmup_epochs:
            # update noise_prior
            self.noisy_prior = self.class_size_noisy / self.num_training_samples
            # update distilled_weight and distilled_label
            self.distilled_weight = self.distilled_weight_new
            self.distilled_label = self.distilled_label_new
            self.distilled_weight_new = torch.zeros(self.num_training_samples)
            self.distilled_label_new = torch.zeros(self.num_training_samples).long()

            # update the T_mat and P_y_distill
            self.P_y_distill = torch.tensor([torch.sum((self.distilled_label == i) * (self.distilled_weight == 1.0)) for i in range(self.num_classes)]).float()
            self.P_y_distill /= torch.sum(self.P_y_distill)
            self.T_mat = torch.zeros((self.num_classes, self.num_classes, self.num_training_samples)).float().to(self.device)
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    # https://github.com/UCSC-REAL/CAL/blob/bdb376bb0300616937c5369f24350617f495c61b/datamanage/peerdatasets/CIFAR.py#L205
                    self.T_mat[i][j] = ((self.distilled_label == i) * (self.train_labels.to(self.device) == j) * (self.distilled_weight > 0.0)) * 1.0
                    # https://github.com/UCSC-REAL/CAL/blob/bdb376bb0300616937c5369f24350617f495c61b/experiments/exptPeerReg.py#L125-L126
                    weight_sum = max((torch.sum(self.distilled_weight * (self.distilled_label==i)),1.0))                    
                    self.T_mat[i,j] =  (self.T_mat[i,j] - torch.sum(self.T_mat[i,j])/weight_sum) * self.distilled_weight * (self.distilled_label==i)
            # update the criterion to crossentropy_CAL
            self.criterion = PeerLossRegCE(self.alpha, self.noisy_prior, 'crossentropy_CAL', T_mat = self.T_mat.to(self.device), P_y_distill = self.P_y_distill.to(self.device))
            self.alpha_scheduler = self.alpha_scheduler_cls(self.criterion, **self.alpha_scheduler_args)
        else:
            # Don't step the alpha_scheduler immediately after switching it after end of warmup.
            self.alpha_scheduler.step() 

        if self.current_epoch < self.warmup_epochs:
            self.lr_schedulers()[0].step()
        else:
            self.loss_mean_all = self.loss_rec_all / self.T_mat_indicator_sum
            self.lr_schedulers()[1].step()


    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, labels, index = batch[0]

        if self.current_epoch >= self.warmup_epochs:
            outputs = self.model(x)
            distilled_labels = self.distilled_label[index.cpu()].to(self.device)
            distilled_weights = self.distilled_weight[index.cpu()].to(self.device)
            optimizer = self.optimizers()[1]
            optimizer.zero_grad()
            loss, _, _, _, loss_rec_all = self.criterion(outputs, labels, outputs.clone(), distill_y = distilled_labels, raw_idx = index, loss_mean_all = self.loss_mean_all,  distilled_weights = distilled_weights)
            self.loss_rec_all += loss_rec_all
        else:
            outputs = self.model_warmup(x)
            optimizer = self.optimizers()[0]
            optimizer.zero_grad()
            loss, _, _ = self.criterion(outputs, labels, outputs.clone())

        preds = torch.max(outputs, 1)[1] 
        if self.current_epoch < self.warmup_epochs:
            out_prob = softmax(outputs, dim=-1)
            probs_label = out_prob.gather(1, (labels).view(-1,1)).view(-1)
            log_out_prob = -torch.log(out_prob + EPS)
            ce_term = -torch.log(probs_label + EPS)
            norm_term = torch.mean(log_out_prob, dim=1) 

            # distill (sieve) samples (general case)
            idx_true = (ce_term - norm_term < -8.0).detach()
            idx_false = (ce_term - norm_term > -8.0).detach()
            self.distilled_weight_new[index[idx_true]] = 1.0
            self.distilled_weight_new[index[idx_false]] = 1.0
            self.distilled_label_new[index[idx_true]] = labels[idx_true]
            self.distilled_label_new[index[idx_false]] = preds[idx_false]
        
        loss.backward()
        optimizer.step()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(preds, labels), on_epoch=True, on_step=False)
        return loss
    

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_true = batch
        if self.current_epoch >= self.warmup_epochs:
            y_pred = self.model(x)
        else:
            y_pred = self.model_warmup(x)
        self.val_acc(y_pred, y_true)
        loss = self.val_criterion(y_pred, y_true)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        if self.current_epoch >= self.warmup_epochs:
            y_pred = self.model(x)
        else:
            y_pred = self.model_warmup(x)
        self.test_acc(y_pred, y)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer_warmup = self.optimizer_cls(self.model_warmup.parameters(), **self.optimizer_args)
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_args)
        scheduler_warmup = self.scheduler_cls(optimizer_warmup, **self.scheduler_args)
        return [optimizer_warmup, optimizer], [scheduler_warmup, scheduler]