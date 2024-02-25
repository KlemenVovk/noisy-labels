from lightning import Trainer
from aim.pytorch_lightning import AimLogger

from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR

from methods.classifiers.resnet import resnet34

from methods.learning_strategies.GLS.GLS import GLS
from methods.learning_strategies.GLS.utils import ResNet34 # They are using their own resnet

from configs.base import MethodConfig
from configs.data.cifar10 import cifar10_base_config

from data.pipelines.noise.noises import SymmetricNoise
from data.pipelines.noise.pipeline import AddNoise

lr_plan_warmup = [0.1] * 40 + [0.01] * 40 + [0.001] * 40
lr_plan_main = [0.1] * 100 + [0.01] * 50 + [0.001] * 50

class cifar10_noise(cifar10_base_config):
    dataset_train_augmentation = AddNoise(SymmetricNoise(10, 0.6))

class positive_ls_cifar10_noise(MethodConfig):
    data_config = cifar10_noise

    # classifier = resnet34
    # classifier_args = dict(
    #     weights=None,
    #     num_classes=10
    # )

    classifier = ResNet34
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = GLS
    learning_strategy_args = {
        "smooth_rate": 0.4, # has to be negative
        "warmup_epochs": 120,
        # "warmup_epochs": 5,
    }

    optimizer_cls = SGD
    optimizer_args = [dict(lr=1, momentum=0.9, weight_decay=1e-4, nesterov=True), dict(lr=1, momentum=0.9, weight_decay=1e-4, nesterov=True)]
    scheduler_cls = LambdaLR
    scheduler_args = [dict(lr_lambda=lambda epoch: lr_plan_warmup[epoch]), dict(lr_lambda=lambda epoch: lr_plan_main[epoch - len(lr_plan_warmup)])]


    trainer_args = dict(
        max_epochs=120+200,
        # max_epochs=5+5,
        deterministic=True,
        logger=AimLogger(experiment="pls")
    )

    seed = 1337