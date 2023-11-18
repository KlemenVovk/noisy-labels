from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.models.resnet import resnet34
import lightning as L
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from utils.cores2 import loss_cores, f_beta

# TODO: LR scheduling should be more flexible (current model works only for exactly 100 epochs like the authors proposed).
# TODO: implement cores2* - the second phase from the paper (consistency training).
# TODO: log the warmup!
# TODO: maybe it would be good to log the accuracy with the noisy labels as a sanity check to see if the loss is working and making a difference (because it can happen that a NN is just powerful enough to work even with a bad/baseline loss)

'''ResNet in PyTorch.
TAKEN FROM: https://github.com/UCSC-REAL/cores/blob/main/models/resnet.py

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3],num_classes=num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3],num_classes=num_classes)

def ResNet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3],num_classes=num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3],num_classes=num_classes)


# First phase from the paper: https://arxiv.org/abs/2010.02347
# Uses resnet34 as the backbone (not pretrained). Trained with CORES loss.
# Basically works on priors of the label noise and iteratively updates the priors after each epoch.
class SampleSieve(L.LightningModule):
    def __init__(self, initial_lr, momentum, weight_decay, use_author_resnet, datamodule):
        super().__init__()
        # saves arguments (hyperparameters) passed to the constructor as self.hparams and logs them to hparams.yaml.
        self.save_hyperparameters(ignore=["datamodule"])
        self.num_training_samples = datamodule.num_training_samples
        self.num_classes = datamodule.num_classes
        self._compute_initial_noise_prior(datamodule)
        self.use_author_resnet = use_author_resnet
        if self.use_author_resnet:
            self.model = ResNet34(num_classes=self.num_classes)
        else:
            self.model = resnet34(weights=None, num_classes=self.num_classes) # don't use pretrained weights
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, top_k=1, task='multiclass')
        self.noisy_class_frequency = torch.tensor([0] * self.num_classes).cuda()
        
    def _compute_initial_noise_prior(self, datamodule):
        # Noise prior is just the class probabilities
        class_frequency = torch.tensor([0] * self.num_classes)
        for y in datamodule.train_dataset.noisy_targets:
            class_frequency[y] += 1
        self.initial_noise_prior = class_frequency / class_frequency.sum()
        self.initial_noise_prior = self.initial_noise_prior.cuda()
        self.cur_noise_prior = self.initial_noise_prior


    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y_noisy, y_true = batch
        logits = self.model(x)        
        # clean_indicators is a list of 0s and 1s, where 1 means that the label is "predicted" to be clean, 0 means that the label is "predicted" to be noisy
        loss, clean_indicators = loss_cores(self.current_epoch, logits, y_noisy, noise_prior=self.cur_noise_prior)
        self.train_acc(logits, y_true)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Record the frequency of predicted noisy classes
        for i, clean_indicator in enumerate(clean_indicators):
            if clean_indicator == 0:
                self.noisy_class_frequency[y_noisy[i]] += 1
        return loss

    def on_train_epoch_end(self):
        # Once the training epoch is done, update our prior of the label noise.
        self.cur_noise_prior = self.initial_noise_prior*self.num_training_samples - self.noisy_class_frequency
        self.cur_noise_prior = self.cur_noise_prior/sum(self.cur_noise_prior)
        self.cur_noise_prior = self.cur_noise_prior.cuda()
        self.noisy_class_frequency = torch.tensor([0] * self.num_classes).cuda()

    
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)
        loss, _ = loss_cores(self.current_epoch, logits, y, noise_prior=self.cur_noise_prior)
        self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # Here multiple optimizers and schedulers can be set. Currently we have hardcoded the lr scheduling to exactly like it is in the paper.
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.initial_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        lr_plan = [0.1] * 50 + [0.01] * (50 + 1) # +1 because lr is set before the check if the training should stop due to reaching max_epochs (so its updated at the end of each epoch, for the next epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_plan[epoch]/(1+f_beta(epoch)))
        return [optimizer], [scheduler]