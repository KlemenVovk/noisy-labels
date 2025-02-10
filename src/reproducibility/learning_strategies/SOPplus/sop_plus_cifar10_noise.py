from torch.optim import SGD
from pytorch_lightning.loggers import CSVLogger

from noisypy.data.pipelines.base import Compose
from noisypy.data.pipelines.index import AddIndex
from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.noise.noises import BiasedSymmetricNoise
from noisypy.data.pipelines.double_augmentation import DoubleAugmentation


from .sop_plus_cifar10_clean import (
    sop_plus_cifar10_clean, cifar10_double_augmentation_index_config,
    transform1, transform2)

class cifar10_double_augmentation_index_noise_config(cifar10_double_augmentation_index_config):

    dataset_train_augmentation = Compose([
        AddNoise(noise=BiasedSymmetricNoise(num_classes=10, noise_rate=0.5)),
        AddIndex(),
        DoubleAugmentation(transform1=transform1, transform2=transform2)])

class sop_plus_cifar10_noise(sop_plus_cifar10_clean):

    data_config = cifar10_double_augmentation_index_noise_config

    learning_strategy_args = dict(
        ratio_consistency = 0.9,
        ratio_balance = 0.1,
        lr_u = 10,
        lr_v = 10,
        overparam_mean = 0.0,
        overparam_std = 1e-8,
        overparam_momentum = 0,
        overparam_weight_decay = 0,
        overparam_optimizer_cls = SGD
    )

    trainer_args = dict(
        max_epochs=300,
        deterministic=True,
        logger=CSVLogger("../logs", name="sop_plus_symmetric")
    )

    seed = 1337