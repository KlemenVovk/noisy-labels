from pytorch_lightning.loggers import CSVLogger
from torch.optim import Adam

from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.noise.noises import SymmetricNoise
from .pes_cifar10_clean import pes_cifar10_clean

from ..common import cifar10_base_config


class pes_cifar10_noisy_data_config(cifar10_base_config):
    dataset_train_augmentation = AddNoise(noise=SymmetricNoise(num_classes=10, noise_rate=0.5))


class pes_cifar10_noise(pes_cifar10_clean):

    data_config = pes_cifar10_noisy_data_config

    learning_strategy_args = dict(
        PES_lr=1e-4,
        warmup_epochs=25,
        T2=7,
        T3=5,
        optimizer_refine_cls= Adam,
    )

    trainer_args = dict(
        reload_dataloaders_every_n_epochs=1,
        max_epochs=200,
        deterministic=True,
        logger=CSVLogger("../logs", name="pes_symmetric")
    )

    seed = 123