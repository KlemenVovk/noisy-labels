from typing import Any, Type
from copy import deepcopy

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torchmetrics

from .utils import CrossEntropyLossStable, ProbLossStable, Divergence
from ..base import LearningStrategyWithWarmupModule


class FDivergence(LearningStrategyWithWarmupModule):

    def __init__(self,
                 datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: Type[Optimizer], optimizer_args: dict,
                 scheduler_cls: Type[LRScheduler], scheduler_args: dict,
                 warmup_epochs: int,
                 divergence: str = "Total-Variation",
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args, 
            optimizer_cls, optimizer_args, scheduler_cls,
            scheduler_args, warmup_epochs, *args, **kwargs)
        
        # manual optimization
        self.automatic_optimization = False
        
        # model
        self.model = classifier_cls(**classifier_args)
        
        # criterions and divergence
        self.divergence = Divergence(name=divergence)
        self.criterion = CrossEntropyLossStable()
        self.criterion_prob = ProbLossStable()
        
        # metrics
        self.num_classes = datamodule.num_classes
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        
        # misc
        self.warmup_epochs = warmup_epochs - 1 if warmup_epochs > 1 else 0
        self.num_train_samples = datamodule.num_train_samples
        self.num_val_samples = datamodule.num_val_samples
        self.num_test_samples = datamodule.num_test_samples
        
        # best model
        self.best_warmup_model = None
        self.best_model = None
        # best f_score on the validation set (fscore is basically -loss on the validation set so idk why we need to maximize this instead of minimizing the loss)
        self.max_f_score = -100
        # best accuracy on the validation set (idk why the authors call it like this)
        self.best_prob_acc = 0
    
    
    def warmup_training_step(self, batch: Any, batch_idx: int, *args) -> STEP_OUTPUT:
        opt = self.optimizers()[0]
        opt.zero_grad()
        
        (x, y), _, _ = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        self.manual_backward(loss)
        opt.step()
        
        acc = self.train_acc(output, y)
        return loss, acc
    
    
    def fdivergence_training_step(self, batch: Any, batch_idx: int, *args) -> STEP_OUTPUT:
        opt = self.optimizers()[1]
        opt.zero_grad()
        
        (x_1, y_1_noisy), (x_2, _), (_, y_3_noisy) = batch
        
        output_1 = self.model(x_1)
        prob_reg = -self.criterion_prob(output_1, y_1_noisy)
        loss_regular = self.divergence.activation(prob_reg)
        
        output_2 = self.model(x_2)
        prob_peer = -self.criterion_prob(output_2, y_3_noisy)
        loss_peer = self.divergence.conjugate(prob_peer)
        
        loss = loss_regular - loss_peer
        self.manual_backward(loss)
        opt.step()
        
        acc = self.train_acc(output_1, y_1_noisy)
        return loss, acc


    def training_step(self, batch: Any, batch_idx: int, *args) -> STEP_OUTPUT:
        # calculate loss depending on the stage
        if self.current_epoch <= self.warmup_epochs:
            loss, acc = self.warmup_training_step(batch, batch_idx, *args)
        else:
            loss, acc = self.fdivergence_training_step(batch, batch_idx, *args)
        
        # logging
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, on_step=False)
        return loss
    
    
    def on_train_epoch_end(self):
        # step the scheduler (note that scheduler after warmup starts counting from 0)
        sch = self.lr_schedulers()[0] if self.current_epoch <= self.warmup_epochs else self.lr_schedulers()[1]
        sch.step()
    
    
    def warmup_validation_step(self, batch: Any, batch_idx: int, *args) -> STEP_OUTPUT:
        (x, y), _, _ = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        acc = self.val_acc(output, y)
        
        return loss, acc
    
    
    def fdivergence_validation_step(self, batch: Any, batch_idx: int, *args) -> STEP_OUTPUT:
        (x_1, y_1), (x_2, _), (_, y_3) = batch
        
        output_1 = self.model(x_1)
        prob_reg = - self.criterion_prob(output_1, y_1)
        loss_regular = self.divergence.activation(prob_reg)
        
        output_2 = self.model(x_2)
        prob_peer = - self.criterion_prob(output_2, y_3)
        loss_peer = self.divergence.conjugate(prob_peer)
        
        score = loss_peer - loss_regular
        # the logger automatically accumulates the score so we don't need to multiply by the batch size like the authors do
        # f_score = score.item() * y_1.size(0) 
        f_score = score.item()
        
        acc = self.val_acc(output_1, y_1)
        # NOTE the score is just the loss with switched signs
        # loss = loss_regular - loss_peer
        # return loss, acc, f_score
        return -score, acc, f_score
        
        
    def validation_step(self, batch: Any, batch_idx: int, *args) -> STEP_OUTPUT:        
        if self.current_epoch <= self.warmup_epochs:
            warmup_loss, val_acc = self.warmup_validation_step(batch, batch_idx, *args)
            
            self.log("f_score", -100.0, on_step=False, on_epoch=True)
            
            # log the losses separately since they are computed differently
            self.log("val_loss_warmup", warmup_loss, prog_bar=True)
            self.log("val_loss_fdiv", 100., prog_bar=True)
            # also log the combined curve to compare at the end
            self.log("val_loss", warmup_loss, prog_bar=True)
            
        else:
            f_div_loss, val_acc, f_score = self.fdivergence_validation_step(batch, batch_idx, *args)
            
            self.log("f_score", f_score, on_step=False, on_epoch=True)
            
            # log the losses separately since they are computed differently
            self.log("val_loss_fdiv", f_div_loss, prog_bar=True)
            self.log("val_loss_warmup", 100., prog_bar=True)
            # also log the combined curve to compare at the end
            self.log("val_loss", f_div_loss, prog_bar=True)
        
        self.log("val_acc", val_acc, on_step=False, on_epoch=True)
    
    def on_validation_epoch_end(self) -> None:
        if self.current_epoch <= self.warmup_epochs:
            # in the warmup validation save the best model based on the validation accuracy
            val_acc = self.trainer.callback_metrics["val_acc"]
            if val_acc > self.best_prob_acc:
                self.best_warmup_model = deepcopy(self.model)
                self.best_prob_acc = val_acc
            
            # at the end of warmup switch the model to the best one
            if self.current_epoch == self.warmup_epochs:
                self.model.load_state_dict(self.best_warmup_model.state_dict())
        else:
            # in the f-divergence validation save the best model based on the f_score
            f_score = self.trainer.callback_metrics["f_score"]
            if f_score > self.max_f_score:
                self.best_model = deepcopy(self.model)
                self.max_f_score = f_score
    
    
    def configure_optimizers(self):
        optim_warmup = self.optimizer_cls[0](params=self.model.parameters(), **self.optimizer_args[0])
        optim_fdivergence = self.optimizer_cls[1](params=self.model.parameters(), **self.optimizer_args[1])
        
        scheduler_warmup = self.scheduler_cls[0](optim_warmup, **self.scheduler_args[0])
        scheduler_fdivergence = self.scheduler_cls[1](optim_fdivergence, **self.scheduler_args[1])
        
        return [optim_warmup, optim_fdivergence], [scheduler_warmup, scheduler_fdivergence]