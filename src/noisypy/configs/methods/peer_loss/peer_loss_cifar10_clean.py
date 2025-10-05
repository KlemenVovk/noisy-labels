from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch.loggers import CSVLogger

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.peer_loss.utils import lr_plan
from noisypy.methods.learning_strategies.peer_loss.peer_loss import PeerLoss
from noisypy.data.pipelines.base import Identity
from noisypy.data.pipelines.shuffle import ShuffleImages
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config


class cifar10_clean_shuffle_config(cifar10_base_config):
    dataset_train_augmentation = [
        Identity(),
        ShuffleImages(),
    ]


stages = [0, 50]


class peer_loss_cifar10_clean_config(MethodConfig):
    data_config = cifar10_clean_shuffle_config

    classifier = resnet34
    classifier_args = dict(
        num_classes=10,
        weights=None,
    )

    learning_strategy_cls = PeerLoss
    learning_strategy_args = dict(stage_epochs=stages)

    optimizer_cls = [SGD, SGD]
    optimizer_args = [
        dict(lr=1, weight_decay=1e-4),
        dict(lr=1, weight_decay=1e-4),
    ]

    scheduler_cls = [LambdaLR, LambdaLR]
    scheduler_args = [
        dict(lr_lambda=lambda e: lr_plan(e)),
        dict(lr_lambda=lambda e: lr_plan(e + stages[0])),
    ]

    trainer_args = dict(
        max_epochs=sum(stages) + 1,
        deterministic=True,
        logger=CSVLogger("../logs", name="peer_loss_cifar10_clean"),
    )

    seed = 1337
