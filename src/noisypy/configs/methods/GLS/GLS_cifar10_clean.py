from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from lightning.pytorch.loggers import CSVLogger

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.GLS.GLS import GLS
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config


class GLS_cifar10_clean_config(MethodConfig):
    data_config = cifar10_base_config

    classifier = resnet34
    classifier_args = dict(weights=None, num_classes=10)

    learning_strategy_cls = GLS
    learning_strategy_args = dict(
        warmup_epochs=2,
        smooth_rate=-0.2,
    )

    optimizer_cls = SGD
    optimizer_args = [dict(lr=0.01, momentum=0.9, weight_decay=5e-4)] * 2
    scheduler_cls = MultiStepLR
    scheduler_args = [dict(milestones=[60], gamma=0.1)] * 2

    trainer_args = dict(
        max_epochs=100,
        deterministic=True,
        logger=CSVLogger("../logs", name="GLS_cifar10_clean"),
    )

    seed = 1337
