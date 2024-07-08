from noisypy.data.datasets.cifar10 import CIFAR10, cifar10_test_transform, cifar10_train_transform
from noisypy.configs.base.data import DataConfig
from noisypy.data.datasets.cifar10n import CIFAR10N

class cifar10_base_config(DataConfig):

    dataset_train_cls = dataset_val_cls = dataset_test_cls = CIFAR10
    
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
    dataset_test_args = dict(
        train=False,
        transform=cifar10_test_transform
    )

    datamodule_args = dict(
        batch_size=128,
        num_workers=2
    )


class cifar10n_clean_config(DataConfig):

    dataset_train_cls = dataset_val_cls = dataset_test_cls = CIFAR10N
    dataset_args = dict(
        noise_type="clean_label",
        noise_dir="../data/noisylabels",
        cifar_dir="../data/cifar",
        download=True,
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

class cifar10n_aggre_config(cifar10n_clean_config):

    dataset_args = dict(
        noise_type="aggre_label",
        noise_dir="../data/noisylabels",
        cifar_dir="../data/cifar"
    )

class cifar10n_worse_config(cifar10n_clean_config):

    dataset_args = dict(
        noise_type="worse_label",
        noise_dir="../data/noisylabels",
        cifar_dir="../data/cifar"
    )

class cifar10n_random1_config(cifar10n_clean_config):

    dataset_args = dict(
        noise_type="random_label1",
        noise_dir="../data/noisylabels",
        cifar_dir="../data/cifar"
    )

class cifar10n_random2_config(cifar10n_clean_config):

    dataset_args = dict(
        noise_type="random_label2",
        noise_dir="../data/noisylabels",
        cifar_dir="../data/cifar"
    )

class cifar10n_random3_config(cifar10n_clean_config):

    dataset_args = dict(
        noise_type="random_label3",
        noise_dir="../data/noisylabels",
        cifar_dir="../data/cifar"
    )