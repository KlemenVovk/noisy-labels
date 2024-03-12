from aim.pytorch_lightning import AimLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.FBT.FBT import BackwardT
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config


class backwardT_cifar10_clean(MethodConfig):

    data_config = cifar10_base_config

    classifier = resnet34
    classifier_args = dict(
        weights=None,
        num_classes=10
    )

    learning_strategy_cls = BackwardT
    learning_strategy_args = dict(
        warmup_epochs=0,
        filter_outliers=False,
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[60], gamma=0.1)

    trainer_args = dict(
        max_epochs=240,
        deterministic=True,
        num_sanity_val_steps=0,
        logger=AimLogger(experiment="backwardT")
    )

    seed = 1337