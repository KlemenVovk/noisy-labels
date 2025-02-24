from pytorch_lightning.loggers import CSVLogger

from noisypy.data.pipelines.base import Compose
from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.noise.noises import BiasedSymmetricNoise
from noisypy.data.pipelines.divide_mix import DivideMixify
from .divide_mix_cifar10_clean import (
    divide_mix_cifar10_base_config,
    divide_mix_cifar10_clean,
)


noise_rate = 0.5


class divide_mix_cifar10_noise_config(divide_mix_cifar10_base_config):
    dataset_train_augmentation = Compose(
        [
            AddNoise(noise=BiasedSymmetricNoise(noise_rate=noise_rate, num_classes=10)),
            DivideMixify(),
        ]
    )


class divide_mix_cifar10_noise(divide_mix_cifar10_clean):
    data_config = divide_mix_cifar10_noise_config

    learning_strategy_args = dict(
        warmup_epochs=10,
        noise_type="symmetric",
        noise_rate=noise_rate,
        p_thresh=0.5,
        temperature=0.5,
        alpha=4,
    )

    trainer_args = dict(
        max_epochs=300,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        logger=CSVLogger("../logs", name="divide_mix_symmetric"),
    )

    seed = 1337
