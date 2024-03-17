from aim.pytorch_lightning import AimLogger

from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from methods.classifiers.resnet import resnet34
from methods.learning_strategies.CE.CE import CE
from configs.base.method import MethodConfig
from configs.data.cifar10 import cifar10_base_config, CIFAR10
from data.pipelines.split import Split


class cifar10_split(cifar10_base_config):

    dataset_train_cls, dataset_val_cls, dataset_test_cls = (*Split(0.8)(CIFAR10), CIFAR10)
    dataset_val_args = {**cifar10_base_config.dataset_val_args, "train":True}


class CE_cifar10_split(MethodConfig):

    data_config = cifar10_split

    classifier = resnet34
    classifier_args = dict(
        weights=None,
        num_classes=10
    )

    learning_strategy_cls = CE

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[60], gamma=0.1)

    trainer_args = dict(
        max_epochs=100,
        deterministic=True,
        logger=AimLogger(experiment="CE")
    )

    seed = 1337