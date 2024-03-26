from typing import Any

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torchmetrics
import torch
from torch.nn import Module
from torch.nn.functional import cross_entropy, one_hot
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..base import LearningStrategyModule
from .utils import elr_plus_loss, sigmoid_rampup


class ELR_plus(LearningStrategyModule):
    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 beta: float, lmbd: float, gamma: float, alpha: float, 
                 ema_update: bool, ema_step: int, coef_step: float,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args,
            optimizer_cls, optimizer_args, 
            scheduler_cls, scheduler_args, *args, **kwargs)
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        # beta, lmbd, gamma, alpha, ema_update, ema_step, coef_step
        self.save_hyperparameters(ignore=["classifier_cls", "classifier_args", "datamodule", 
                                          "optimizer_cls", "optimizer_args", 
                                          "scheduler_cls", "scheduler_args"])
        self.num_training_samples = datamodule.num_train_samples
        self.num_batches = len(datamodule.train_dataloader()[0])
        self.num_classes = datamodule.num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.ema_update = ema_update
        self.ema_step = ema_step

        # init model
        self.model1 = classifier_cls(**classifier_args)
        self.model2 = classifier_cls(**classifier_args)
        self.model_ema1 = classifier_cls(**classifier_args)
        self.model_ema2 = classifier_cls(**classifier_args)
        self.criterion1 = elr_plus_loss(self.num_training_samples, num_classes=self.num_classes, 
                                        lmbd=lmbd, beta=beta, coef_step=coef_step)
        self.criterion2 = elr_plus_loss(self.num_training_samples, num_classes=self.num_classes, 
                                        lmbd=lmbd, beta=beta, coef_step=coef_step)
        self.val_criterion = cross_entropy

        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        
        self.automatic_optimization = False


    def update_ema_variables(self, model: Module, model_ema: Module, global_step: int) -> None:
        # Use the true average until the exponential average is more correct
        # dropping the first if statement as it appears to be a bug
        # https://github.com/shengliu66/ELR/blob/934af53434a336b6db80d05d7649d23216e8ca6d/ELR_plus/trainer/trainer.py#L323-L324
        if self.ema_update:
            alpha = sigmoid_rampup(global_step + 1, self.ema_step) * self.gamma
        else:
            alpha = min(1 - 1 / (global_step + 1), self.gamma)
        for ema_param, param in zip(model_ema.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float=1.0) -> Any:
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
            lam = max(lam, 1-lam)
            batch_size = x.size()[0]
            mix_index = torch.randperm(batch_size)

            mixed_x = lam * x + (1 - lam) * x[mix_index, :]
            mixed_target = lam * y + (1 - lam) * y[mix_index, :]

            return mixed_x, mixed_target, lam, mix_index
        else:
            lam = 1
            return x, y, lam, ...
        

    def train_step(self, batch: Any, model: Module, model_ema1: Module, 
        model_ema2: Module, criterion: Module, opt: Optimizer) -> STEP_OUTPUT:
        model.train()
        model_ema1.train()

        x, y_noise, index = batch
        x_original = x
        y = one_hot(y_noise, 10).float()
        x, y, mixup_l, mix_index = self.mixup_data(x, y,  alpha = self.alpha)

        y_pred = model(x)
        with torch.no_grad():
            y_pred_original = model_ema2(x_original)
        criterion.update_hist(y_pred_original, index.cpu().numpy().tolist(), mix_index = mix_index, mixup_l = mixup_l)
        # https://github.com/shengliu66/ELR/blob/934af53434a336b6db80d05d7649d23216e8ca6d/ELR_plus/trainer/trainer.py#L112
        # loss, probs = train_criterion(self.global_step + local_step, output, target)
        step = self.global_step + (-self.current_epoch // 2) * self.num_batches + 1
        loss = criterion(step, y_pred, y)

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        
        # https://github.com/shengliu66/ELR/blob/934af53434a336b6db80d05d7649d23216e8ca6d/ELR_plus/trainer/trainer.py#L120
        # self.update_ema_variables(model, model_ema, self.global_step + local_step, self.config['ema_alpha'])
        # self.global_step + local_step in the originial is equivalent to pytorch_lightning's self.global_step
        # however we are running two epochs for the original one so we need to pass the adjusted step
        self.update_ema_variables(model, model_ema1, step)

        # logging
        self.log("train_loss", loss, prog_bar=True)
        return loss
    

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        opt1, opt2 = self.optimizers()
        if self.current_epoch % 2 == 0:
            return self.train_step(batch[0], self.model1, self.model_ema1, self.model_ema2, self.criterion1, opt1)
        else:
            return self.train_step(batch[0], self.model2, self.model_ema2, self.model_ema1, self.criterion2, opt2)

    
    def on_train_epoch_end(self) -> None:
        scheduler1, scheduler2 = self.lr_schedulers()
        if self.current_epoch % 2 == 0:
            scheduler1.step()
        else:
            scheduler2.step()


    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        self.model1.eval()
        self.model2.eval()
        x, y = batch
        outputs1 = self.model1(x.cuda())
        outputs2 = self.model2(x.cuda())
        outputs = 0.5 * (outputs1 + outputs2)
        self.val_acc(outputs, y)
        loss = self.val_criterion(outputs, y.cuda())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_pred1 = self.model1(x)
        y_pred2 = self.model2(x)
        y_pred = (y_pred1 + y_pred2) / 2
        self.log("test_acc", self.test_acc(y_pred, y))
    

    def configure_optimizers(self) -> list[list[Optimizer], list[LRScheduler]]:
        optimizer1 = self.optimizer_cls(filter(lambda p: p.requires_grad, self.model1.parameters()), **self.optimizer_args)
        optimizer2 = self.optimizer_cls(filter(lambda p: p.requires_grad, self.model2.parameters()), **self.optimizer_args)
        scheduler1 = self.scheduler_cls(optimizer1, **self.scheduler_args)
        scheduler2 = self.scheduler_cls(optimizer2, **self.scheduler_args)
        return [optimizer1, optimizer2], [scheduler1, scheduler2]