from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from lightning.pytorch.loggers import CSVLogger

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.volminnet.volminnet import VolMinNet
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config


class volminnet_cifar10_clean_config(MethodConfig):
    data_config = cifar10_base_config

    classifier = resnet34
    classifier_args = dict(
        num_classes=10,
        weights=None,
    )

    learning_strategy_cls = VolMinNet
    learning_strategy_args = dict(lam=1e-4)

    optimizer_cls = [SGD, SGD]
    optimizer_args = [
        dict(lr=0.01, momentum=0.9, weight_decay=1e-4),
        dict(lr=0.01, momentum=0.9),
    ]

    scheduler_cls = [MultiStepLR, MultiStepLR]
    scheduler_args = [
        dict(milestones=[30, 60], gamma=0.1),
        dict(milestones=[30, 60], gamma=0.1),
    ]

    trainer_args = dict(
        max_epochs=80,
        deterministic=True,
        logger=CSVLogger("../logs", name="volminnet_cifar10_clean"),
    )

    seed = 1337
