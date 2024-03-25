from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from lightning.pytorch.loggers import CSVLogger

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.t_revision.t_revision import TRevision
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config
  

stages = [100, 120, 120]

class t_revision_cifar10_clean_config(MethodConfig):

    data_config = cifar10_base_config

    classifier = resnet34
    classifier_args = dict(
        num_classes=10,
        weights=None,
    )

    learning_strategy_cls = TRevision
    learning_strategy_args = dict(
        stage_epochs=stages
    )

    optimizer_cls = [SGD, SGD, Adam]
    optimizer_args = [
        dict(lr=0.01, weight_decay=1e-4),
        dict(lr=0.01, weight_decay=1e-4, momentum=0.9),
        dict(lr=5e-7, weight_decay=1e-4)
    ]
    
    scheduler_cls = [LambdaLR, MultiStepLR, LambdaLR]
    scheduler_args = [
        dict(lr_lambda=lambda _: 1),
        dict(milestones=[40, 80], gamma=0.1),
        dict(lr_lambda=lambda _: 1),
    ]

    trainer_args = dict(
        max_epochs=sum(stages)+1,
        deterministic=True,
        num_sanity_val_steps=0,
        logger=CSVLogger("../logs", name="t_revision_cifar10_clean")
    )

    seed = 1337