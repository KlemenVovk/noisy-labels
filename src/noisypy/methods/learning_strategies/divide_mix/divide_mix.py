from typing import Any

from tqdm import tqdm
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torchmetrics
import torch
from torch.nn import Module
from torch.nn.functional import one_hot, softmax
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

from ..base import LearningStrategyModule
from .utils import NegEntropy, SemiLoss, BATCH_MAP, end_warmup, set_probabilities, set_predictions


class DivideMix(LearningStrategyModule):
    def __init__(self, datamodule: L.LightningDataModule,
                 classifier_cls: type[Module], classifier_args: dict,
                 optimizer_cls: type[Optimizer], optimizer_args: dict,
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 warmup_epochs: int, noise_type: str, noise_rate: float, 
                 p_thresh: float, temperature: float, alpha: float,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args,
            optimizer_cls, optimizer_args, 
            scheduler_cls, scheduler_args, *args, **kwargs)
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        # warmup_epochs, noise_type, noise_rate, p_thresh, temperature, alpha
        self.save_hyperparameters(ignore=["classifier_cls", "classifier_args", "datamodule", 
                                          "optimizer_cls", "optimizer_args", 
                                          "scheduler_cls", "scheduler_args"])
        self.num_training_samples = datamodule.num_train_samples
        self.num_classes = datamodule.num_classes
        self.warmup_epochs = warmup_epochs
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.p_thresh = p_thresh
        self.temperature = temperature
        self.alpha = alpha

        # init model
        self.model1 = classifier_cls(**classifier_args)
        self.model2 = classifier_cls(**classifier_args)
        
        # init metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass', average="micro")

        # init loss
        self.CEloss = torch.nn.CrossEntropyLoss()
        self.sampleCE = torch.nn.CrossEntropyLoss(reduction='none')
        self.conf_penalty = NegEntropy()
        self.SemiLoss = SemiLoss()
        self.loss_hist1 = []
        self.loss_hist2 = []

        self.automatic_optimization = False
    

    def on_train_epoch_start(self) -> None:
        if self.current_epoch >= self.warmup_epochs:
            data_loader = self.trainer.datamodule.train_dataloader()
            prob1, self.loss_hist1 = self.eval_train(self.model1, self.loss_hist1, data_loader, 1)
            prob2, self.loss_hist2 = self.eval_train(self.model2, self.loss_hist2, data_loader, 2)

            pred1 = (prob1 > self.p_thresh)
            pred2 = (prob2 > self.p_thresh)

            # set the second model's predictions for the first model's data and vice versa
            # https://github.com/LiJunnan1992/DivideMix/blob/d9d3058fa69a952463b896f84730378cdee6ec39/Train_cifar.py#L267C1-L273C107
            set_probabilities(self.trainer.datamodule, prob2, prob1)
            set_predictions(self.trainer.datamodule, pred2, pred1)


    def on_train_epoch_end(self) -> None:
        if self.current_epoch + 1 == self.warmup_epochs:
            end_warmup(self.trainer.datamodule)
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()


    def warmup_step(self, batch: Any, model: Module, opt: Optimizer, batch_idx: int) -> torch.Tensor:
        # based on https://github.com/LiJunnan1992/DivideMix/blob/d9d3058fa69a952463b896f84730378cdee6ec39/Train_cifar.py#L124
        model.train()
        x, y_noisy, _ = batch
        opt.zero_grad()
        outputs = model(x.to(self.device))
        loss = self.CEloss(outputs, y_noisy.to(self.device))
        # penalize confident prediction for asymmetric noise
        # https://github.com/LiJunnan1992/DivideMix/blob/d9d3058fa69a952463b896f84730378cdee6ec39/Train_cifar.py#L132
        if self.noise_type == 'asymmetric':     # TODO: This seems like data leakage
            penalty = self.conf_penalty(outputs)
            loss += penalty
        
        # compute gradient and do SGD step
        self.manual_backward(loss)
        opt.step()

        return loss
    

    def eval_train(self, model: Module, loss_hist: list, data_loader: DataLoader, i: int) -> torch.Tensor:
        # based on https://github.com/LiJunnan1992/DivideMix/blob/d9d3058fa69a952463b896f84730378cdee6ec39/Train_cifar.py#L165
        model.eval()
        losses = torch.zeros(self.num_training_samples).to(self.device)
        with torch.no_grad():
            for batch in tqdm(data_loader[BATCH_MAP["eval_train"]], desc=f'Label Guessing {i}', leave=False):
                x, y_noisy, index = batch
                outputs = model(x.to(self.device))
                loss = self.sampleCE(outputs, y_noisy.to(self.device))
                for b in range(x.size(0)):
                    losses[index[b]]=loss[b]    
                index += x.size(0)
        # normalize the losses
        losses = (losses - torch.min(losses)) / (torch.max(losses) - torch.min(losses))
        loss_hist.append(losses)

        # average loss over last 5 epochs to improve convergence stability
        # https://github.com/LiJunnan1992/DivideMix/blob/d9d3058fa69a952463b896f84730378cdee6ec39/Train_cifar.py#L178
        if self.noise_rate == 0.9:              # TODO: This seems like data leakage
            history = torch.stack(loss_hist)
            input_loss = history[-5].mean(0)
            input_loss = input_loss.reshape(-1, 1)
        else:
            input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
        gmm.fit(input_loss.cpu())
        prob = gmm.predict_proba(input_loss.cpu())
        prob = prob[:, gmm.means_.argmin()]
        return prob, loss_hist
    

    def train_step(self, batch_labeled: Any, batch_unlabeled: Any, model: Module, 
                   fixed_model: Module, opt: Optimizer, batch_idx: int, num_iter: int) -> torch.Tensor:
        # based on https://github.com/LiJunnan1992/DivideMix/blob/d9d3058fa69a952463b896f84730378cdee6ec39/Train_cifar.py#L41
        model.train()
        fixed_model.eval()
        inputs_u, inputs_u2 = batch_unlabeled
        inputs_x, inputs_x2, labels_x, w_x = batch_labeled
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = one_hot(labels_x, num_classes=self.num_classes)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor).to(self.device)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = model(inputs_u)
            outputs_u12 = model(inputs_u2)
            outputs_u21 = fixed_model(inputs_u)
            outputs_u22 = fixed_model(inputs_u2)            
            
            pu = (softmax(outputs_u11, dim=1) + softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/self.temperature) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = model(inputs_x)
            outputs_x2 = model(inputs_x2)            
            
            px = (softmax(outputs_x, dim=1) + softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/self.temperature) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()     
        
        # mixmatch
        l = torch.distributions.beta.Beta(self.alpha, self.alpha).sample().item()
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = model(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:] 

        Lx, Lu, lamb = self.SemiLoss(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], self.current_epoch+batch_idx/num_iter, self.warmup_epochs)

        # regularization
        prior = torch.ones(self.num_classes)/self.num_classes
        prior = prior.to(self.device)        
        pred_mean = softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        return loss


    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        opt1, opt2 = self.optimizers()
        if self.current_epoch < self.warmup_epochs:
            batch1 = batch[BATCH_MAP["warmup1"]]
            batch2 = batch[BATCH_MAP["warmup2"]]
            loss1 = self.warmup_step(batch1, self.model1, opt1, batch_idx)
            loss2 = self.warmup_step(batch2, self.model2, opt2, batch_idx)
        else:
            labeled_batch1 = batch[BATCH_MAP["labeled1"]]
            unlabeled_batch1 = batch[BATCH_MAP["unlabeled1"]]
            labeled_batch2 = batch[BATCH_MAP["labeled2"]]
            unlabeled_batch2 = batch[BATCH_MAP["unlabeled2"]]
            num_iter1 = len(self.trainer.datamodule.train_datasets[BATCH_MAP["labeled1"]]) // self.trainer.datamodule.batch_size
            num_iter2 = len(self.trainer.datamodule.train_datasets[BATCH_MAP["labeled2"]]) // self.trainer.datamodule.batch_size
            loss1 = self.train_step(labeled_batch1, unlabeled_batch1, self.model1, self.model2, opt1, batch_idx, num_iter1)
            loss2 = self.train_step(labeled_batch2, unlabeled_batch2, self.model2, self.model1, opt2, batch_idx, num_iter2)

        # average loss over two models
        loss = loss1 + loss2 / 2    
        self.log('train_loss', loss, prog_bar=True)
        return loss

    
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        self.model1.eval()
        self.model2.eval()
        x, y = batch
        outputs1 = self.model1(x.to(self.device))
        outputs2 = self.model2(x.to(self.device))
        outputs = (outputs1 + outputs2) / 2
        self.val_acc(outputs, y)
        loss = self.CEloss(outputs, y.to(self.device))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch: Any, batch_idx: int):
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