from lightning import Trainer
from aim.pytorch_lightning import AimLogger

from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from methods.classifiers.resnet import resnet34
from methods.classifiers.glsresnet import ResNet34
from methods.learning_strategies.negative_ls.NegativeLS import NegativeLS

from configs.base import MethodConfig
from configs.data.cifar10 import cifar10_base_config

lr_plan_main = [0.1] * 100 + [0.01] * 50 + [0.001] * 50
lr_plan_warmup = [0.1] * 40 + [0.01] * 40 + [0.001] * 40
lr_plan = lr_plan_warmup + lr_plan_main

class negative_ls_cifar10_clean(MethodConfig):

    data_config = cifar10_base_config

    # classifier = resnet34
    # classifier_args = dict(
    #     weights=None,
    #     num_classes=10
    # )

    classifier = ResNet34
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = NegativeLS
    learning_strategy_args = {
        "smooth_rate": -0.6, # has to be negative
        "warmup_epochs": 120,
    }

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler_cls = LambdaLR
    scheduler_args = dict(lr_lambda=lambda epoch: lr_plan[epoch])

    trainer_args = dict(
        max_epochs=200,
        deterministic=True,
        logger=AimLogger(experiment="pls")
    )

    seed = 1337