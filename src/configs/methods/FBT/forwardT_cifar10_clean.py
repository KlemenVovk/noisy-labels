from aim.pytorch_lightning import AimLogger

from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from methods.classifiers.resnet import resnet34
from methods.learning_strategies.FBT.FBT import ForwardT

from configs.base.method import MethodConfig
from configs.data.cifar10 import cifar10_base_config

class forwardT_cifar10_clean(MethodConfig):

    data_config = cifar10_base_config

    classifier = resnet34
    classifier_args = dict(
        weights=None,
        num_classes=10
    )

    learning_strategy_cls = ForwardT
    learning_strategy_args = dict(
        warmup_epochs=0
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[60], gamma=0.1)

    trainer_args = dict(
        max_epochs=200,
        deterministic=True,
        logger=AimLogger(experiment="forwardT")
    )

    seed = 1337