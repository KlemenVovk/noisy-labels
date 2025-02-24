from pytorch_lightning.loggers import CSVLogger
from torch.optim import Adam

from .pes_semi_cifar10_clean import pes_semi_cifar10n_clean
from ..common import cifar10n_aggre_config


class pes_semi_cifar10n_noise(pes_semi_cifar10n_clean):
    data_config = cifar10n_aggre_config

    learning_strategy_args = dict(
        PES_lr=1e-4,
        warmup_epochs=20,
        T2=5,
        lambda_u=5,
        temperature=0.5,
        alpha=4,
        optimizer_refine_cls=Adam,
    )

    trainer_args = dict(
        reload_dataloaders_every_n_epochs=1,
        max_epochs=300,
        deterministic=True,
        logger=CSVLogger("../logs", name="pes_semi_noise"),
    )

    seed = 1337
