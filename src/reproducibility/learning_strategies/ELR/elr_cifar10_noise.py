from pytorch_lightning.loggers import CSVLogger

from noisypy.data.pipelines.base import Compose
from noisypy.data.pipelines.index import AddIndex
from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.noise.noises import BiasedSymmetricNoise
from noisypy.configs.data.cifar10 import cifar10_base_config
from .elr_cifar10_clean import elr_cifar10_clean

class cifar10_index_noise_config(cifar10_base_config):
    dataset_train_augmentation = Compose([
        AddNoise(noise=BiasedSymmetricNoise(num_classes=10, noise_rate=0.8)), 
        AddIndex()])

class elr_cifar10_noise(elr_cifar10_clean):

    data_config = cifar10_index_noise_config

    learning_strategy_args = dict(beta = 0.7, lmbd=3)  # β ∈ {0.5, 0.7, 0.9, 0.99}, λ ∈ {1, 3, 5, 7, 10}

    trainer_args = dict(
        max_epochs=150,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="elr_symmetric")
    )

    seed = 1337