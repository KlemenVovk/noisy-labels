from pytorch_lightning.loggers import CSVLogger
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.PES.pes import PES
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config


class PES_cifar10_clean_config(MethodConfig):

    data_config = cifar10_base_config

    classifier = resnet34
    classifier_args = dict(
        num_classes=10,
        weights=None,
    )

    learning_strategy_cls = PES
    learning_strategy_args = dict(
        PES_lr=1e-4,
        T1=25,
        T2=7,
        T3=5,
        optimizer_refine_cls= Adam,
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[100, 150], gamma=0.1)

    trainer_args = dict(
        reload_dataloaders_every_n_epochs=1,
        max_epochs=200,
        deterministic=True,
        logger=CSVLogger("../logs", name="PES_cifar10_clean")
    )

    seed = 1337