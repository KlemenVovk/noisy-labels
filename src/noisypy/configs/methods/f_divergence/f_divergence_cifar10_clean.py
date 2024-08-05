from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from lightning.pytorch.loggers import CSVLogger

from noisypy.data.pipelines.base import Identity
from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.f_divergence.f_divergence import FDivergence
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config


class cifar10_3sample_config(cifar10_base_config):

    dataset_train_augmentation = [Identity()] * 3
    dataset_val_augmentation   = [Identity()] * 3


class FDivergence_cifar10_clean_config(MethodConfig):

    data_config = cifar10_3sample_config

    classifier = resnet34
    classifier_args = dict(
        weights=None,
        num_classes=10
    )

    learning_strategy_cls = FDivergence
    learning_strategy_args = dict(
        warmup_epochs=2,
        divergence="Total-Variation"
    )

    optimizer_cls = [SGD, SGD]
    optimizer_args = [
        dict(lr=0.1, momentum=0.9, weight_decay=5e-4), # warmup stage
        dict(lr=0.01, momentum=0.9, weight_decay=5e-4) # f-divergence stage
    ]
    scheduler_cls = [MultiStepLR, MultiStepLR]
    scheduler_args = [
        dict(milestones=[60, 120, 180, 240], gamma=0.1), # warmup stage
        dict(milestones=[30, 60, 90, 120], gamma=0.1) # f-divergence stage
    ]

    trainer_args = dict(
        max_epochs=100,
        deterministic=True,
        logger=CSVLogger("../logs", name="f_divergence_cifar10_clean"),
    )

    seed = 1337