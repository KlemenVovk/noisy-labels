from typing import Any, Type
import math

from tqdm import tqdm
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torchmetrics
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ..base import LearningStrategyModule
from .utils import (NegEntropy, linear_rampup, co_divide, debias_output, 
                    bias_initial, debias_pl, bias_update, BATCH_MAP, end_warmup, 
                    CE_Soft_Label, ProMixModel)
from .fmix import FMix

class ProMix(LearningStrategyModule):
    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: Type[Module], classifier_args: dict,
                 optimizer_cls: Type[Optimizer], optimizer_args: dict,
                 scheduler_cls: Type[LRScheduler], scheduler_args: dict,
                 warmup_epochs: int, rampup_epochs: int, noise_type: str, 
                 rho_start: float, rho_end: float, debias_beta_pl: float,
                 alpha_output: float, tau: float, start_expand: int,
                 threshold: float, temperature: float,
                 model_type: str, feat_dim: int,
                 bias_m: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args,
            optimizer_cls, optimizer_args, 
            scheduler_cls, scheduler_args, *args, **kwargs)
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters(ignore=["classifier_cls", "classifier_args", "datamodule", 
                                          "optimizer_cls", "optimizer_args", 
                                          "scheduler_cls", "scheduler_args"])
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.noise_type = noise_type
        self.rho_start = rho_start  
        self.rho_end = rho_end
        self.debias_beta_pl = debias_beta_pl
        self.beta = 1
        self.alpha_output = alpha_output
        self.tau = tau
        self.start_expand = start_expand
        self.threshold = threshold
        self.temperature = temperature
        self.bias_m = bias_m
        self.batch_size = self.datamodule.train_dataloader()[BATCH_MAP['train']].batch_size

        self.fmix = FMix()

        self.pi1 = bias_initial(self.num_classes)
        self.pi2 = bias_initial(self.num_classes)
        self.pi1_unrel = bias_initial(self.num_classes)
        self.pi2_unrel = bias_initial(self.num_classes)


        # init model
        self.model1 = ProMixModel(base_model_cls=classifier_cls, model_type=model_type, feat_dim=feat_dim, **classifier_args)
        self.model2 = ProMixModel(base_model_cls=classifier_cls, model_type=model_type, feat_dim=feat_dim, **classifier_args)
        
        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")

        # init loss
        self.CEloss = torch.nn.CrossEntropyLoss()
        self.sampleCE = torch.nn.CrossEntropyLoss(reduction='none')
        self.conf_penalty = NegEntropy()
        self.CESoft = CE_Soft_Label()

        self.automatic_optimization = False
    

    def on_train_epoch_start(self) -> None:
        if self.current_epoch >= self.warmup_epochs:
            data_loader = self.trainer.datamodule.train_dataloader()
            self.w = linear_rampup(self.current_epoch, self.rampup_epochs)
            rho = self.rho_start + (self.rho_end - self.rho_start) * self.w
            prob1 = self.eval_train(self.model1, data_loader, 1, rho)
            prob2 = self.eval_train(self.model2, data_loader, 2, rho)

            # https://github.com/Justherozen/ProMix/blob/40d8378193f1a098473479ebcb56dbafa89dfacb/Train_promix.py#L517-L518
            co_divide(self.trainer.datamodule, prob1, prob2)
            if self.debias_beta_pl:
                self.beta = 0.1 * linear_rampup(self.current_epoch, 2*self.rampup_epochs)
            

    def on_train_epoch_end(self) -> None:
        if self.current_epoch + 1 == self.warmup_epochs:
            end_warmup(self.trainer.datamodule)

        # step the schedulers
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()


    def warmup_step(self, batch: Any, model: Module, opt: Optimizer) -> Tensor:
        # based on https://github.com/Justherozen/ProMix/blob/40d8378193f1a098473479ebcb56dbafa89dfacb/Train_promix.py#L339
        model.train()
        x, y_noisy, _ = batch
        opt.zero_grad()
        outputs = model(x.to(self.device))
        loss = self.CEloss(outputs, y_noisy.to(self.device))
        
        if self.noise_type == 'asymmetric':     # NOTE: This seems like data leakage
            penalty = self.conf_penalty(outputs)
            loss += penalty
        
        # compute gradient and do optimization step
        self.manual_backward(loss)
        opt.step()

        return loss
    

    def eval_train(self, model: Module, data_loader: DataLoader, i: int, rho: float) -> Tensor:
        # based on https://github.com/Justherozen/ProMix/blob/40d8378193f1a098473479ebcb56dbafa89dfacb/Train_promix.py#L419
        model.eval()
        losses = torch.zeros(self.num_training_samples).to(self.device)
        targets_list = torch.zeros(self.num_training_samples).to(self.device)
        
        with torch.no_grad():
            for batch in tqdm(data_loader[BATCH_MAP["eval_train"]], desc=f'Eval Train {i}', leave=False):
                x, y_noisy, index = batch
                outputs = model(x.to(self.device))
                loss = self.sampleCE(outputs, y_noisy.to(self.device))
                for b in range(x.size(0)):
                    losses[index[b]]=loss[b]  
                    targets_list[index[b]] = y_noisy[b]  

        
        #class-wise small-loss selection (CSS for base selection set)
        losses = (losses - torch.min(losses)) / (torch.max(losses) - torch.min(losses))
        
        prob = torch.zeros(targets_list.shape[0])
        idx_chosen_sm = []
        min_len = 1e10
        for j in range(self.num_classes):
            indices = torch.where(targets_list==j)[0]
            if len(indices) == 0:
                continue
            bs_j = targets_list.shape[0] * (1. / self.num_classes)
            pseudo_loss_vec_j = losses[indices]
            sorted_idx_j = pseudo_loss_vec_j.sort()[1]
            partition_j = max(min(int(math.ceil(bs_j*rho)), len(indices)), 1)
            idx_chosen_sm.append(indices[sorted_idx_j[:partition_j]])
            min_len = min(min_len, partition_j)
        idx_chosen_sm = torch.cat(idx_chosen_sm)
        prob[idx_chosen_sm] = 1
        return prob
    

    def label_guessing(self, idx_chosen: Tensor, w_x: Tensor, batch_size: int, score1:Tensor, score2:Tensor, match: Tensor) -> Tensor:
        # based on https://github.com/Justherozen/ProMix/blob/40d8378193f1a098473479ebcb56dbafa89dfacb/Train_promix.py#L88-L89
        w_x2 = w_x.clone()
        # when clean data is insufficient, try to incorporate more examples
        if (1. * idx_chosen.shape[0] / batch_size) < self.threshold:
            # both networks agree
            high_conf_cond2 = (score1 > self.tau) * (score2 > self.tau) * match
            # remove already selected examples; newly selected
            high_conf_cond2 = (1. * high_conf_cond2 - w_x.squeeze()) > 0     
            hc2_idx = torch.where(high_conf_cond2)[0]
            
            # maximally select (batch_size * threshold); where (idx_chosen.shape[0]) selected already
            max_to_sel_num = int(batch_size * self.threshold) - idx_chosen.shape[0]
            
            if high_conf_cond2.sum() > max_to_sel_num:
                # to many examples selected, remove some low conf examples
                score_mean = (score1 + score2) / 2
                idx_remove = (-score_mean[hc2_idx]).sort()[1][max_to_sel_num:]
                high_conf_cond2[hc2_idx[idx_remove]] = False
            w_x2[high_conf_cond2] = 1
        return w_x2


    def mixup(self, net: Module, inputs_x: Tensor, idx_chosen: Tensor, pseudo_label_l: Tensor, pi: Tensor, use_ph: bool = False) -> tuple[Tensor, Tensor]:
        # based on https://github.com/Justherozen/ProMix/blob/40d8378193f1a098473479ebcb56dbafa89dfacb/Train_promix.py#L204-L238 and
        # https://github.com/Justherozen/ProMix/blob/40d8378193f1a098473479ebcb56dbafa89dfacb/Train_promix.py#L265-L299
        l = torch.distributions.beta.Beta(4, 4).sample().item()
        l = max(l, 1-l)
        X_w_c = inputs_x[idx_chosen]
        pseudo_label_c = pseudo_label_l[idx_chosen]
        idx = torch.randperm(X_w_c.size(0))
        X_w_c_rand = X_w_c[idx]
        pseudo_label_c_rand = pseudo_label_c[idx]
        X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand
        pseudo_label_c_mix = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        if use_ph:
            _, logits_mix = net(X_w_c_mix, use_ph=use_ph)
        else:
            logits_mix = net(X_w_c_mix, use_ph=use_ph)
        logits_mix = debias_output(logits_mix, pi, self.alpha_output)
        loss_mix = self.CESoft(logits_mix, targets=pseudo_label_c_mix).mean()
        x_fmix = self.fmix(X_w_c)
        if use_ph:
            _, logits_fmix = net(x_fmix, use_ph=use_ph)
        else:
            logits_fmix = net(x_fmix, use_ph=use_ph)
        logits_fmix = debias_output(logits_fmix, pi, self.alpha_output)
        loss_fmix = self.fmix.loss(logits_fmix, (pseudo_label_c.detach()).long())
        return loss_mix, loss_fmix


    def loss(self, outputs_x: Tensor, outputs_x2: Tensor, outputs_x_ph: Tensor, outputs_x2_ph: Tensor, idx_chosen: Tensor, 
             idx_unchosen: Tensor, pseudo_label_l: Tensor,  debias_px_unrel: Tensor, outputs_x_unrel_ph: Tensor, 
             outputs_x2_unrel_ph: Tensor, loss_mix: Tensor, loss_fmix: Tensor, loss_mix_ph: Tensor, loss_fmix_ph: Tensor) -> tuple[Tensor, Tensor]:
        # based on https://github.com/Justherozen/ProMix/blob/40d8378193f1a098473479ebcb56dbafa89dfacb/Train_promix.py#L240-L260 and
        # https://github.com/Justherozen/ProMix/blob/40d8378193f1a098473479ebcb56dbafa89dfacb/Train_promix.py#L300-L317
        loss_cr = self.CESoft(outputs_x2[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()
        loss_cr_ph = self.CESoft(outputs_x2_ph[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()

        # cross entropy loss for primary head and pseudo head
        loss_ce = self.CESoft(outputs_x[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()
        loss_ce_ph = self.CESoft(outputs_x_ph[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()
        # loss for net1-primary head
        loss_net1 = loss_ce + self.w * (loss_cr + loss_mix + loss_fmix)

        # loss for noisy samples on the pseudo head 
        ptx = debias_px_unrel ** (1 / self.temperature)
        ptx = ptx / ptx.sum(dim=1, keepdim=True)
        beta = 0 if (self.current_epoch >= 2*self.rampup_epochs and self.beta < 1) else self.beta
        targets_urel = ptx
        loss_unrel_ph = self.CESoft(outputs_x_unrel_ph[idx_unchosen], targets=targets_urel[idx_unchosen]).mean()
        loss_unrel_ph += self.w * self.CESoft(outputs_x2_unrel_ph[idx_unchosen], targets=targets_urel[idx_unchosen]).mean()
        #loss for net1-pseudo head
        loss_net1_ph = beta * loss_unrel_ph + loss_ce_ph + self.w * (loss_cr_ph + loss_mix_ph + loss_fmix_ph)
        return loss_net1, loss_net1_ph


    def train_step(self, batch: Any, opt1: Optimizer, opt2: Optimizer) -> Tensor:
        # based on https://github.com/Justherozen/ProMix/blob/40d8378193f1a098473479ebcb56dbafa89dfacb/Train_promix.py#L110
        self.model1.train()
        self.model2.train() # train two peer networks in parallel
        inputs_x, inputs_x2, labels_x, w_x, w_x2 = batch
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = one_hot(labels_x, num_classes=self.num_classes)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor).to(self.device)
        w_x2 = w_x2.view(-1, 1).type(torch.FloatTensor).to(self.device)

        # inputs_x: weak augmentation
        # inputs_x2: strong augmentation
        outputs_x, outputs_x_ph, _ = self.model1(inputs_x,train=True,use_ph=True)
        outputs_x2, outputs_x2_ph, _ = self.model1(inputs_x2,train=True,use_ph=True)
        outputs_a, outputs_a_ph, _ = self.model2(inputs_x,train=True,use_ph=True)
        outputs_a2, outputs_a2_ph, _ = self.model2(inputs_x2,train=True,use_ph=True)
        outputs_x_ori = outputs_x.clone().detach()
        outputs_a_ori = outputs_a.clone().detach()        

        # debiasing logit for Debiased Margin-based Loss calculation of reliable samples D_l on primary head
        outputs_x = debias_output(outputs_x, self.pi1, self.alpha_output)
        outputs_x2 = debias_output(outputs_x2, self.pi1, self.alpha_output)
        outputs_a = debias_output(outputs_a, self.pi2, self.alpha_output)
        outputs_a2 = debias_output(outputs_a2, self.pi2, self.alpha_output)

        # debiasing logit for Debiased Margin-based Loss calculation of unreliable samples D_u on pseudo head
        outputs_x_unrel_ph = debias_output(outputs_x_ph, self.pi1_unrel, self.alpha_output)
        outputs_x2_unrel_ph = debias_output(outputs_x2_ph, self.pi1_unrel, self.alpha_output)
        outputs_a_unrel_ph = debias_output(outputs_a_ph, self.pi2_unrel, self.alpha_output)
        outputs_a2_unrel_ph = debias_output(outputs_a2_ph, self.pi2_unrel, self.alpha_output)

        # debiasing logit for Debiased Margin-based Loss calculation of reliable samples D_u on pseudo head
        outputs_x_ph = debias_output(outputs_x_ph, self.pi1, self.alpha_output)
        outputs_x2_ph = debias_output(outputs_x2_ph, self.pi1, self.alpha_output)
        outputs_a_ph = debias_output(outputs_a_ph, self.pi2, self.alpha_output)
        outputs_a2_ph = debias_output(outputs_a2_ph, self.pi2, self.alpha_output)

        with torch.no_grad():
            # original p, stored for distribution estimation
            px = torch.softmax(outputs_x_ori, dim=1)
            px2 = torch.softmax(outputs_a_ori, dim=1)

            # debiasing for the generation of pseudo-labels
            debias_px = debias_pl(outputs_x_ori, self.pi1, self.debias_beta_pl)
            debias_px2 = debias_pl(outputs_a_ori, self.pi2, self.debias_beta_pl)
            debias_px_unrel = debias_pl(outputs_x_ori, self.pi1_unrel, self.debias_beta_pl)
            debias_px2_unrel = debias_pl(outputs_a_ori, self.pi2_unrel, self.debias_beta_pl)

            #one-hot label for the samples selected by label guessing (LGA) 
            pred_net = one_hot(debias_px.max(dim=1)[1], self.num_classes).float()
            pred_net2 = one_hot(debias_px2.max(dim=1)[1], self.num_classes).float()

            # matched high-confidence selection (MHCS)
            high_conf_cond = (labels_x * px).sum(dim=1) > self.tau
            high_conf_cond2 = (labels_x * px2).sum(dim=1) > self.tau
            w_x[high_conf_cond] = 1
            w_x2[high_conf_cond2] = 1

            #For CSS&MHCS: adopt original label; For LGA: adopt predicted label
            pseudo_label_l = labels_x * w_x + pred_net * (1 - w_x)
            pseudo_label_l2 = labels_x * w_x2 + pred_net2 * (1 - w_x2)

            idx_chosen = torch.where(w_x == 1)[0]
            idx_unchosen = torch.where(w_x != 1)[0]
            idx_chosen_2 = torch.where(w_x2 == 1)[0]
            idx_unchosen_2 = torch.where(w_x2 != 1)[0]

            # label guessing by agreement (LGA) for last K epochs
            if self.current_epoch > self.trainer.max_epochs - self.start_expand:
                score1 = px.max(dim=1)[0]
                score2 = px2.max(dim=1)[0]
                match = px.max(dim=1)[1] == px2.max(dim=1)[1]
                hc2_sel_wx1 = self.label_guessing(idx_chosen, w_x, batch_size, score1, score2, match)
                hc2_sel_wx2 = self.label_guessing(idx_chosen_2, w_x2, batch_size, score1, score2, match)
                idx_chosen = torch.where(hc2_sel_wx1 == 1)[0]
                idx_chosen_2 = torch.where(hc2_sel_wx2 == 1)[0]
                idx_unchosen = torch.where(hc2_sel_wx1 != 1)[0]
                idx_unchosen_2 = torch.where(hc2_sel_wx2 != 1)[0]

        # mixup loss for primary head $h$ of Net 1; adopt vanilla mixup and fmix: https://github.com/ecs-vlc/FMix
        loss_mix, loss_fmix = self.mixup(self.model1, inputs_x, idx_chosen, pseudo_label_l, self.pi1)

        # mixup loss for pseudo head $h_{AP}$ of Net 1
        loss_mix_ph, loss_fmix_ph = self.mixup(self.model1, inputs_x, idx_chosen, pseudo_label_l, self.pi1, use_ph=True)

        # compute loss for net1
        loss_net1, loss_net1_ph = self.loss(
            outputs_x, 
            outputs_x2, 
            outputs_x_ph, 
            outputs_x2_ph, 
            idx_chosen, 
            idx_unchosen,
            pseudo_label_l,  
            debias_px_unrel, 
            outputs_x_unrel_ph, 
            outputs_x2_unrel_ph,
            loss_mix, 
            loss_fmix, 
            loss_mix_ph, 
            loss_fmix_ph
        )
        
        # mixup loss for primary head $h$ of Net 2
        loss_mix2, loss_fmix2 = self.mixup(self.model2, inputs_x, idx_chosen_2, pseudo_label_l2, self.pi2)

        # mixup loss for pseudo head of Net 2
        loss_mix_ph2, loss_fmix_ph2 = self.mixup(self.model2, inputs_x, idx_chosen_2, pseudo_label_l2, self.pi2, use_ph=True)

        # compute loss for net2
        loss_net2, loss_net2_ph = self.loss(
            outputs_a,
            outputs_a2,
            outputs_a_ph,
            outputs_a2_ph,
            idx_chosen_2,
            idx_unchosen_2,
            pseudo_label_l2,
            debias_px2_unrel,
            outputs_a_unrel_ph,
            outputs_a2_unrel_ph,
            loss_mix2,
            loss_fmix2,
            loss_mix_ph2,
            loss_fmix_ph2
        )

        # total loss
        loss = loss_net1 + loss_net2 + loss_net1_ph + loss_net2_ph
        # moving average estimation of bias for D_l and D_u seperately
        self.pi1 = bias_update(px[idx_chosen], self.pi1, self.bias_m)
        self.pi2 = bias_update(px2[idx_chosen_2], self.pi2, self.bias_m)
        self.pi1_unrel = bias_update(px[idx_unchosen], self.pi1_unrel, self.bias_m)
        self.pi2_unrel = bias_update(px2[idx_unchosen_2],self. pi2_unrel, self.bias_m)
            
        # compute gradient and do optimization step
        opt1.zero_grad()
        opt2.zero_grad()
        self.manual_backward(loss)
        opt1.step()
        opt2.step()

        return loss, loss_net1, loss_net2, loss_net1_ph, loss_net2_ph


    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:

        opt1, opt2 = self.optimizers()
        if self.current_epoch < self.warmup_epochs:
            batch = batch[BATCH_MAP["warmup"]]
            loss1 = self.warmup_step(batch, self.model1, opt1)
            loss2 = self.warmup_step(batch, self.model2, opt2)
            loss = (loss1 + loss2) / 2    
        else:
            batch = batch[BATCH_MAP['train']]
            if batch[0].size(0) < self.batch_size:
                # This is equivalent to using drop_last but keeps eval_train dataloader with drop_last=False
                # (https://github.com/Justherozen/ProMix/blob/40d8378193f1a098473479ebcb56dbafa89dfacb/dataloader_cifarn.py#L279)
                return 
            loss, loss_net1, loss_net2, loss_net1_ph, loss_net2_ph = self.train_step(batch, opt1, opt2)
            # NOTE: Could also log the individual losses
            loss1 = (loss_net1 + loss_net1_ph) / 2
            loss2 = (loss_net2 + loss_net2_ph) / 2

        self.log('loss1', loss1, prog_bar=True)
        self.log('loss2', loss2, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss
   

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        self.model1.eval()
        self.model2.eval()
        x, y = batch
        y_pred1 = self.model1(x)
        y_pred2 = self.model2(x)
        y_pred = (y_pred1 + y_pred2) / 2
        self.val_acc(y_pred, y)
        loss = self.CEloss(y_pred, y.to(self.device))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    

    def test_step(self, batch: Any, batch_idx: int):
        self.model1.eval()
        self.model2.eval()
        x, y = batch
        y_pred1 = self.model1(x)
        y_pred2 = self.model2(x)
        y_pred = (y_pred1 + y_pred2) / 2
        self.test_acc(y_pred, y)
        self.log("test_acc", self.test_acc, on_epoch=True)


    def configure_optimizers(self) -> list[list[Optimizer], list[LRScheduler]]:
        optimizer1 = self.optimizer_cls(self.model1.parameters(), **self.optimizer_args)
        optimizer2 = self.optimizer_cls(self.model2.parameters(), **self.optimizer_args) 
        scheduler1 = self.scheduler_cls(optimizer1, **self.scheduler_args)
        scheduler2 = self.scheduler_cls(optimizer2, **self.scheduler_args)
        return [optimizer1, optimizer2], [scheduler1, scheduler2]