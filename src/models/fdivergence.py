from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.models.resnet import resnet34
import lightning as L
import torch
import torchmetrics
from models.resnet_fdiv import resnet_cifar34
from utils.fdivergence import CrossEntropyLossStable, ProbLossStable, Divergence


class FDivergence(L.LightningModule):
    def __init__(self, initial_lr, momentum, weight_decay, datamodule, divergence, warmup_epochs):
        super().__init__()
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters(ignore=["datamodule"])
        self.datamodule = datamodule
        self.num_training_samples = datamodule.num_training_samples
        self.num_classes = datamodule.num_classes
        # TODO: enable setting an arbitrary backbone model
        self.model = resnet34(weights=None, num_classes=self.num_classes)  # don't use pretrained weights
        
        # from original paper:
        # self.model = resnet_cifar34(num_classes=self.num_classes)
        # NOTE: they use a pre activation resnet (modification on the order of layers in resnet)
        
        # metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task="multiclass")
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task="multiclass")
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task="multiclass")
        
        self.divergence = Divergence(divergence)
        self.criterion = CrossEntropyLossStable()
        self.criterion_prob = ProbLossStable()
        self.warmup_epochs = warmup_epochs
        self.warmup = False

    def _f_divergence_value(self, y_noisy, y_noisy_2, y_output, y_output_1):
    # y_noisy: noisy label of the 1st sample
    # y_noisy_2: noisy label of the 3rd sample
    #
    # y_output: output of x for the 1st sample
    # y_output_1: output of x for the 2nd sample
        prob_reg = - self.criterion_prob(y_output, y_noisy)
        loss_regular = self.divergence.activation(prob_reg)
        
        prob_peer = - self.criterion_prob(y_output_1, y_noisy_2)
        loss_peer = self.divergence.conjugate(prob_peer)
        loss = loss_regular - loss_peer
        
        # TODO
        # score = loss
        # f_score += score * target.size(0)
        # return f_score/10000
    
        return loss
    
    def on_train_epoch_start(self):
        if self.current_epoch < self.warmup_epochs:
            self.warmup = True
        else:
            self.warmup = False
    
    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:        
        (x_1, y_1_noisy, _), (x_2, _, _), (_, y_3_noisy, _) = batch
        
        # pass the first sample through the model
        logits_1 = self.model(x_1)
        
        # TODO:
        # 1) add training options with bias (needs noise estimation)
        
        if self.warmup:
            # regular CE with noisy labels (NOTE: their implementation of CE)
            loss = self.criterion(logits_1, y_1_noisy)
        else:
            # loss on the 1st sample
            prob_reg = - self.criterion_prob(logits_1, y_1_noisy)
            loss_regular = self.divergence.activation(prob_reg)
            
            # loss on the feature of the 2nd sample and the noisy label of the 3rd sample
            logits_2 = self.model(x_2)
            prob_peer = - self.criterion_prob(logits_2, y_3_noisy)
            loss_peer = self.divergence.conjugate(prob_peer)
            
            # combine losses
            loss = loss_regular - loss_peer
        
        # train accuracy is calculated via NOISY training data
        self.train_acc(logits_1, y_1_noisy)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noisy, y_true = batch
        
        logits = self.model(x)
        loss = self.criterion(logits, y_noisy)
        
        # Validation acc is calculated via NOISY validation data
        self.val_acc(logits, y_noisy)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # TODO
        # can't use test_step while training and didn't find a workaround
        # just calculate the test accuracy here as for cifar we have the same validation and test set for now
        loss_test = self.criterion(logits, y_true)
        self.test_acc(logits, y_true)
        self.log("test_loss", loss_test, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    
    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_true = batch
        
        logits = self.model(x)
        loss = self.criterion(logits, y_true)
        
        self.test_acc(logits, y_true)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Here multiple optimizers and schedulers can be set. Currently we have hardcoded the lr scheduling to exactly like it is in the paper.
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.initial_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        
        # TODO: possibility to set lr_plan easier
        
        # lr from original paper
        lr_plan = [1e-2] * 30 + [1e-3] * 30 + [1e-4] * 30 + [1e-5] * 30 + [1e-6] * (30 + 1)  # +1 because lr is set before the check if the training should stop due to reaching max_epochs (so its updated at the end of each epoch, for the next epoch)
        
        # lr from noisylabels
        # lr_plan = [0.1] * 50 + [0.01] * (50 + 1) # +1 because lr is set before the check if the training should stop due to reaching max_epochs (so its updated at the end of each epoch, for the next epoch)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_plan[epoch])
        return [optimizer], [scheduler]


# Create a custom callback to trigger testing every N epochs
# class TestEveryNEpochs(L.Callback):
#     def __init__(self, test_interval=10):  # Change test_interval to your desired frequency
#         super().__init__()
#         self.test_interval = test_interval

#     def on_validation_epoch_end(self, trainer, pl_module):
#         if (trainer.current_epoch + 1) % self.test_interval == 0:
#             trainer.test(pl_module, pl_module.datamodule)
