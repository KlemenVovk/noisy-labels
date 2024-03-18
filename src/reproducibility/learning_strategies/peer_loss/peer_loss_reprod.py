from aim.pytorch_lightning import AimLogger
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from noisypy.methods.learning_strategies.peer_loss.utils import lr_plan
from .utils import resnet_cifar18_pre
from noisypy.methods.learning_strategies.peer_loss.peer_loss import PeerLoss
from noisypy.data.pipelines.base import Compose
from noisypy.data.pipelines.noise.noises import SymmetricNoise, InstanceNoise
from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.shuffle import ShuffleImages
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config

from pathlib import Path
noise = SymmetricNoise(10, 0.2)
#noise = InstanceNoise(torch.load(Path(__file__).parent.parent.parent.parent / "methods/learning_strategies/peer_loss/reproducibility/original_noise.pt"))
#noise = InstanceNoise(torch.load("methods/learning_strategies/peer_loss/reproducibility/original_noise.pt"))

class cifar10_noise(cifar10_base_config):

    dataset_train_augmentation = [
        AddNoise(noise),
        Compose([ShuffleImages(), AddNoise(noise)])
    ]

stages = [0, 50]

class peer_loss_reprod(MethodConfig):

    data_config = cifar10_noise

    classifier = resnet_cifar18_pre
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = PeerLoss
    learning_strategy_args = dict(
        stage_epochs=stages
    )

    optimizer_cls = [SGD, SGD]
    optimizer_args = [
        dict(lr=1, weight_decay=1e-4),
        dict(lr=1, weight_decay=1e-4),
    ]
    
    scheduler_cls = [LambdaLR, LambdaLR]
    scheduler_args = [
        dict(lr_lambda=lambda e: lr_plan(e)),
        dict(lr_lambda=lambda e: lr_plan(e+stages[0])),
    ]

    trainer_args = dict(
        max_epochs=sum(stages)+1,
        deterministic=True,
        logger=AimLogger(experiment="PeerLoss")
    )

    seed = 1337