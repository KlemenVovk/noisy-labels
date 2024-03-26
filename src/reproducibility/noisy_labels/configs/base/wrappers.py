from noisypy.data.pipelines import AddIndex, DivideMixify, Identity, ShuffleImages, Compose, DoubleAugmentation
from noisypy.data.datasets.cifar10 import cifar10_train_transform
from noisypy.methods.learning_strategies.SOPplus.utils import autoaug_paper_cifar10


# some serious mental retardation incoming
def add_index_wrapper(data_config):
    class new_data_config(data_config):
        dataset_train_augmentation = AddIndex()

    return new_data_config

def dividemixify_wrapper(data_config):
    class new_data_config(data_config):
        dataset_train_args = [
            dict(mode="all",        transform=cifar10_train_transform),
            dict(mode="all",        transform=cifar10_train_transform),
            dict(mode="unlabeled",  transform=cifar10_train_transform),
            dict(mode="unlabeled",  transform=cifar10_train_transform),
            dict(mode="all",        transform=cifar10_train_transform),
        ]
        dataset_train_augmentation = DivideMixify()

    return new_data_config

def peer_wrapper(data_config):
    class new_data_config(data_config):
        dataset_train_augmentation = [
            Identity(),
            ShuffleImages(),
        ]
    
    return new_data_config

def double_aug_wrapper(data_config):
    class new_data_config(data_config):
        dataset_train_args = {**data_config.dataset_train_args, "transform": None}
        dataset_train_augmentation = Compose(
            [AddIndex(), DoubleAugmentation(transform1=cifar10_train_transform, transform2=autoaug_paper_cifar10)]
        )
    
    return new_data_config