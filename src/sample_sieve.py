import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from models.resnet import ResNet34
from data.cifar import CIFAR10DataModule

# TODO: aim logger

model = ResNet34(10)
trainer = L.Trainer(max_epochs=1)
dm = CIFAR10DataModule(train_transform=transforms.ToTensor(), add_synthetic_noise=True)
trainer.fit(model, dm)