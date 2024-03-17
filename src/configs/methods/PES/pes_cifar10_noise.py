from pytorch_lightning.loggers import CSVLogger
from torch.optim import Adam

from data.pipelines.noise.pipeline import AddNoise
from data.pipelines.noise.noises import SymmetricNoise

from configs.methods.PES.pes_cifar10_clean import pes_cifar10_clean
from configs.data.cifar10 import cifar10_base_config


class pes_cifar10_noisy_data_config(cifar10_base_config):
    dataset_train_augmentation = AddNoise(noise=SymmetricNoise(num_classes=10, noise_rate=0.5))


class pes_cifar10_noise(pes_cifar10_clean):

    data_config = pes_cifar10_noisy_data_config

    learning_strategy_args = dict(
        PES_lr=1e-4,
        T1=25,
        T2=7,
        T3=5,
        optimizer_refine_cls= Adam,
    )

    trainer_args = dict(
        reload_dataloaders_every_n_epochs=1,
        max_epochs=200,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="pes_symmetric")
    )

    seed = 123