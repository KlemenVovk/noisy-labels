from pytorch_lightning.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.DISC.DISC import DISC
from noisypy.methods.learning_strategies.DISC.utils import (
    weak_train_transform,
    strong_train_transform,
)
from noisypy.data.pipelines.base import Compose
from noisypy.data.pipelines.index import AddIndex
from noisypy.data.pipelines.double_augmentation import DoubleAugmentation
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config


class cifar10_clean_DISC_double_aug(cifar10_base_config):
    dataset_train_args = dict(transform=None)

    dataset_train_augmentation = Compose(
        [
            AddIndex(),
            DoubleAugmentation(weak_train_transform, strong_train_transform),
        ]
    )

    datamodule_args = dict(
        batch_size=128,
        num_workers=0,
    )


class DISC_cifar10_clean_config(MethodConfig):
    data_config = cifar10_clean_DISC_double_aug

    classifier = resnet34
    classifier_args = dict(num_classes=10)

    learning_strategy_cls = DISC
    learning_strategy_args = dict(
        start_epoch=1,
        alpha=5.0,  # 5.0
        sigma=0.5,  # 0.5
        momentum=0.95,  # 0.95
        lambd_ce=1.0,  # 1
        lambd_h=1.0,  # 1
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0, weight_decay=0.001)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[80, 160])

    trainer_args = dict(
        max_epochs=200,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        logger=CSVLogger("../logs", name="disc_cifar10_clean"),
    )

    seed = 1337
