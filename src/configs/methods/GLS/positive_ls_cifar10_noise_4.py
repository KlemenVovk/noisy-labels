from lightning import Trainer
from aim.pytorch_lightning import AimLogger

from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR

from methods.classifiers.resnet import resnet34

from methods.learning_strategies.GLS.GLS import GLS
from methods.learning_strategies.GLS.utils import ResNet34 # They are using their own resnet
from methods.learning_strategies.GLS.cifar10 import lambda_gls_noise, GLSCIFAR10 # They have their own take on SymmetricNoise

from configs.base import MethodConfig
from configs.data.cifar10 import cifar10_base_config

from data.pipelines.noise.noises import SymmetricNoise, LambdaNoise
from data.pipelines.noise.pipeline import AddNoise

# LR plan when warmup is used
lr_plan_warmup = [0.1] * 40 + [0.01] * 40 + [0.001] * 40
lr_plan_main = [1e-6] * 100

# LR plan when warmup is NOT used
# lr_plan = [0.1] * 100 + [0.01] * 50 + [0.001] * 50

class cifar10_noise(cifar10_base_config):
    # dataset_train_augmentation = AddNoise(SymmetricNoise(10, 0.6))
    dataset_cls = GLSCIFAR10
    # TODO: choose the right noise_rate by choosing the right noise file in LambdaNoise
    dataset_train_augmentation = AddNoise(LambdaNoise(lambda_gls_noise))


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
        "smooth_rate": 0.4,
        "warmup_epochs": len(lr_plan_warmup),
    }

    optimizer_cls = SGD
    optimizer_args = [dict(lr=1), dict(lr=1, momentum=0.9, weight_decay=1e-4, nesterov=True)]
    scheduler_cls = LambdaLR
    scheduler_args = [dict(lr_lambda=lambda epoch: lr_plan_warmup[epoch]), dict(lr_lambda=lambda epoch: lr_plan_main[epoch])]

    trainer_args = dict(
        max_epochs=len(lr_plan_warmup) + len(lr_plan_main),
        deterministic=True,
        logger=AimLogger(experiment="pls")
    )

    seed = 3