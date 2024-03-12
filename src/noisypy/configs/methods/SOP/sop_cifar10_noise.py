from torch.optim import SGD
from pytorch_lightning.loggers import CSVLogger

from noisypy.data.pipelines.base import Compose
from noisypy.data.pipelines.index import AddIndex
from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.noise.noises import BiasedSymmetricNoise
from noisypy.configs.data.cifar10 import cifar10_base_config
from .sop_cifar10_clean import sop_cifar10_clean


class cifar10_index_noise_config(cifar10_base_config):

    dataset_train_augmentation = Compose([
        AddNoise(noise=BiasedSymmetricNoise(num_classes=10, noise_rate=0.5)), 
        AddIndex()])

class sop_cifar10_noise(sop_cifar10_clean):

    data_config = cifar10_index_noise_config

    learning_strategy_args = dict(
        ratio_consistency = 0,
        ratio_balance = 0,
        lr_u = 10,
        lr_v = 10,
        overparam_mean = 0.0,
        overparam_std = 1e-8,
        overparam_momentum = 0,
        overparam_weight_decay = 0,
        overparam_optimizer_cls = SGD
    )

    trainer_args = dict(
        max_epochs=120,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="sop_symmetric")
    )

    seed = 1337