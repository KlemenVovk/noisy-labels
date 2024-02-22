from typing import Any

from tqdm import tqdm
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torchmetrics
import torch
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from methods.learning_strategies.base import LearningStrategyModule
from methods.learning_strategies.ELRplus.utils import elr_plus_loss, sigmoid_rampup


class ELR_plus(LearningStrategyModule):
    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args,
            optimizer_cls, optimizer_args, 
            scheduler_cls, scheduler_args, *args)
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        # initial_lr, momentum, weight_decay, beta, lmbd, gamma, alpha, ema_update, ema_step
        self.save_hyperparameters(ignore=["classifier_cls", "classifier_args", "datamodule", 
                                          "optimizer_cls", "optimizer_args", 
                                          "scheduler_cls", "scheduler_args"])
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes

        # init model
        self.model1 = classifier_cls(**classifier_args)
        self.model2 = classifier_cls(**classifier_args)
        self.model_ema1 = classifier_cls(**classifier_args)
        self.model_ema2 = classifier_cls(**classifier_args)
        self.criterion1 = elr_plus_loss(self.num_training_samples, num_classes=self.num_classes, lmbd=self.hparams.lmbd, beta=self.hparams.beta, coef_step=self.hparams.coef_step)
        self.criterion2 = elr_plus_loss(self.num_training_samples, num_classes=self.num_classes, lmbd=self.hparams.lmbd, beta=self.hparams.beta, coef_step=self.hparams.coef_step)
        self.val_criterion = cross_entropy

        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")
    
        self.automatic_optimization = False


    def update_ema_variables(self, model, model_ema, global_step):
        # Use the true average until the exponential average is more correct
        # dropping the first if statement as it appears to be a bug
        # https://github.com/shengliu66/ELR/blob/934af53434a336b6db80d05d7649d23216e8ca6d/ELR_plus/trainer/trainer.py#L323-L324
        if self.hparams.ema_update:
            alpha = sigmoid_rampup(global_step + 1, self.hparams.ema_step)*self.hparams.gamma
        else:
            alpha = min(1 - 1 / (global_step + 1), self.hparams.gamma)
        for ema_param, param in zip(model_ema.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


    def mixup_data(self, x, y, alpha=1.0):
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
        

    def train_step(self, batch: Any, model: torch.nn.Module, model_ema1: torch.nn.Module, 
        model_ema2: torch.nn.Module, criterion: torch.nn.Module, opt: torch.optim.Optimizer) -> STEP_OUTPUT:
        model.train()
        model_ema1.train()
        # model_ema2.eval()
        x, y_noise, index = batch
        x_original = x
        y = torch.zeros(len(y_noise), self.num_classes,).cuda().scatter_(1, y_noise.view(-1,1), 1) 
        x, y, mixup_l, mix_index = self.mixup_data(x, y,  alpha = self.hparams.alpha)

        y_pred = model(x)
        with torch.no_grad():
            y_pred_original = model_ema2(x_original)
        criterion.update_hist(y_pred_original, index.cpu().numpy().tolist(), mix_index = mix_index, mixup_l = mixup_l)
        # loss, probs = train_criterion(self.global_step + local_step, output, target)
        # https://github.com/shengliu66/ELR/blob/934af53434a336b6db80d05d7649d23216e8ca6d/ELR_plus/trainer/trainer.py#L112
        num_batches = len(self.datamodule.train_dataloader()[0])
        step = self.global_step + (-self.current_epoch // 2) * num_batches + 1
        loss, _ = criterion(step, y_pred, y)

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # self.update_ema_variables(model, model_ema, self.global_step + local_step, self.config['ema_alpha'])
        # self.global_step + local_step in the originial is equivalent to pytorch_lightning's self.global_step
        # however we are running two epochs for the original one so we need to pass the adjusted step
        # https://github.com/shengliu66/ELR/blob/934af53434a336b6db80d05d7649d23216e8ca6d/ELR_plus/trainer/trainer.py#L120
        self.update_ema_variables(model, model_ema1, step)
        return loss
    

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        opt1, opt2 = self.optimizers()
        if self.current_epoch % 2 == 0:
            loss = self.train_step(batch[0], self.model1, self.model_ema1, self.model_ema2, self.criterion1, opt1)
        else:
            loss = self.train_step(batch[0], self.model2, self.model_ema2, self.model_ema1, self.criterion2, opt2)
        # logging
        self.log("train_loss", loss, prog_bar=True)
        return loss

    
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
        with torch.no_grad():
            outputs1 = self.model1(x.cuda())
            outputs2 = self.model2(x.cuda())
            outputs = 0.5 * (outputs1 + outputs2)
        self.val_acc(outputs, y)
        loss = self.val_criterion(outputs, y.cuda())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    

    def configure_optimizers(self):
        optimizer1 = self.optimizer_cls(filter(lambda p: p.requires_grad, self.model1.parameters()), **self.optimizer_args)
        optimizer2 = self.optimizer_cls(filter(lambda p: p.requires_grad, self.model2.parameters()), **self.optimizer_args)
        scheduler1 = self.scheduler_cls(optimizer1, **self.scheduler_args)
        scheduler2 = self.scheduler_cls(optimizer2, **self.scheduler_args)
        return [optimizer1, optimizer2], [scheduler1, scheduler2]