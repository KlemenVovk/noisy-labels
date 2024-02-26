from data.datasets.cifar10 import CIFAR10, cifar10_test_transform, cifar10_train_transform

from configs.base.data import DataConfig

class cifar10_base_config(DataConfig):

    dataset_cls = CIFAR10
    dataset_args = dict(
        root="../data/cifar",
        download=True
    )
    dataset_train_args = dict(
        transform=cifar10_train_transform
    )
    dataset_val_args = dict(
        train=False,
        transform=cifar10_test_transform
    )
    dataset_test_args = dataset_val_args

    datamodule_args = dict(
        batch_size=128,
        num_workers=2
    )