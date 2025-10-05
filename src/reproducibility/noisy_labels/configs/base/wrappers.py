from noisypy.configs.base.data import DataConfig
from noisypy.data.pipelines import (
    AddIndex,
    DivideMixify,
    Identity,
    ShuffleImages,
    Compose,
    DoubleAugmentation,
    ProMixify,
)
from noisypy.data.datasets.cifar10 import (
    cifar10_train_transform,
    cifar10_test_transform,
)
from noisypy.methods.learning_strategies.SOPplus.utils import autoaug_paper_cifar10
from noisypy.methods.learning_strategies.DISC.utils import (
    weak_train_transform,
    strong_train_transform,
)


def add_index_wrapper(data_config: DataConfig) -> DataConfig:
    class new_data_config(data_config):
        dataset_train_augmentation = AddIndex()

    return new_data_config


def dividemixify_wrapper(data_config: DataConfig) -> DataConfig:
    class new_data_config(data_config):
        dataset_train_args = [
            dict(mode="all", transform=cifar10_train_transform),
            dict(mode="all", transform=cifar10_train_transform),
            dict(mode="unlabeled", transform=cifar10_train_transform),
            dict(mode="unlabeled", transform=cifar10_train_transform),
            dict(mode="all", transform=cifar10_test_transform),
        ]
        dataset_train_augmentation = DivideMixify()

    return new_data_config


def peer_wrapper(data_config: DataConfig) -> DataConfig:
    class new_data_config(data_config):
        dataset_train_augmentation = [
            Identity(),
            ShuffleImages(),
        ]

    return new_data_config


def double_aug_wrapper(data_config: DataConfig) -> DataConfig:
    class new_data_config(data_config):
        dataset_train_args = {**data_config.dataset_train_args, "transform": None}
        dataset_train_augmentation = Compose(
            [
                AddIndex(),
                DoubleAugmentation(
                    transform1=cifar10_train_transform, transform2=autoaug_paper_cifar10
                ),
            ]
        )

    return new_data_config


def disc_aug_wrapper(data_config):
    class new_data_config(data_config):
        dataset_train_args = {**data_config.dataset_train_args, "transform": None}
        dataset_train_augmentation = Compose(
            [
                AddIndex(),
                DoubleAugmentation(
                    transform1=weak_train_transform, transform2=strong_train_transform
                ),
            ]
        )

    return new_data_config


def promixify_wrapper(data_config: DataConfig) -> DataConfig:
    class new_data_config(data_config):
        dataset_train_args = [
            dict(mode="all", transform=cifar10_train_transform),
            dict(mode="all", transform=cifar10_train_transform),
        ]
        datamodule_args = dict(
            batch_size=256,
            num_workers=2,
        )
        dataset_train_augmentation = ProMixify()

    return new_data_config
