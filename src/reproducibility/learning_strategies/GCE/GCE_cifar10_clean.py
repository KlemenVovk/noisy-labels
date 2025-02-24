from lightning.pytorch.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.GCE.GCE import GCE
from noisypy.data.pipelines.index import AddIndex
from noisypy.configs.base.method import MethodConfig

from ..common import cifar10_base_config


class cifar10_index_config(cifar10_base_config):
    dataset_train_augmentation = AddIndex()


class GCE_cifar10_clean(MethodConfig):
    data_config = cifar10_index_config

    classifier = resnet34
    classifier_args = dict(weights=None, num_classes=10)

    learning_strategy_cls = GCE
    learning_strategy_args = dict(prune_start_epoch=1, prune_freq=10)

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[60], gamma=0.1)

    trainer_args = dict(
        max_epochs=100,
        deterministic=True,
        logger=CSVLogger("../logs", name="GCE_clean"),
    )

    seed = 1337
