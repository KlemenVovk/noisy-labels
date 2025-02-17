from noisypy.data.datasets.cifar10 import (
    cifar10_test_transform,
    cifar10_train_transform,
)
from noisypy.data.datasets.cifar100n import CIFAR100N
from noisypy.data.pipelines.split import Split

from ..base.data import DataConfig


class cifar100n_clean_config(DataConfig):

    dataset_train_cls, dataset_val_cls = Split(1)(CIFAR100N)
    dataset_test_cls = CIFAR100N

    dataset_args = dict(
        noise_type="clean_label",
        noise_dir="../data/noisylabels",
        cifar_dir="../data/cifar",
        download=True,
    )
    dataset_train_args = dict(transform=cifar10_train_transform)
    dataset_val_args = dict(transform=cifar10_test_transform)
    dataset_test_args = dict(train=False, transform=cifar10_test_transform)

    datamodule_args = dict(batch_size=128, num_workers=0)


class cifar100n_noisy_config(cifar100n_clean_config):

    dataset_args = dict(
        noise_type="noisy_label",
        noise_dir="../data/noisylabels",
        cifar_dir="../data/cifar",
        download=True,
    )


# Configs used for the benchmark (with validation set)
class cifar100n_clean_benchmark_config(cifar100n_clean_config):

    dataset_train_cls, dataset_val_cls = Split(0.9)(CIFAR100N)

class cifar100n_noisy_benchmark_config(cifar100n_noisy_config):

    dataset_train_cls, dataset_val_cls = Split(0.9)(CIFAR100N)