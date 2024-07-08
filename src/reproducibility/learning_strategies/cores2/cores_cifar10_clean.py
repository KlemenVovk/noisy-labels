from aim.pytorch_lightning import AimLogger

from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.cores2.cores2 import SampleSieve
from noisypy.methods.learning_strategies.cores2.utils import f_beta
from noisypy.configs.base.method import MethodConfig

from ..common import cifar10_base_config


lr_plan = [0.1] * 50 + [0.01] * (50 + 1)

class cores_cifar10_clean(MethodConfig):

    data_config = cifar10_base_config

    classifier = resnet34
    classifier_args = dict(
        weights=None,
        num_classes=10
    )

    learning_strategy_cls = SampleSieve

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = LambdaLR
    scheduler_args = dict(lr_lambda=lambda epoch: lr_plan[epoch] / (1+f_beta(epoch)))

    trainer_args = dict(
        max_epochs=100,
        deterministic=True,
        logger=AimLogger(experiment="cores2")
    )

    seed = 1337