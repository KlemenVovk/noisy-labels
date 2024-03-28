from noisypy.methods.learning_strategies.PES.pes import PES
from .base.config import NoisyLabelsMethod, CSVLogger
from torch.optim import Adam


class PES_config(NoisyLabelsMethod):

    learning_strategy_cls = PES
    learning_strategy_args = dict(
        PES_lr=1e-4,
        warmup_epochs=25,
        T2=7,
        T3=5,
        optimizer_refine_cls=Adam,
    )

    trainer_args = dict(
        reload_dataloaders_every_n_epochs=1,
        max_epochs=100,
        deterministic=True,
        logger=CSVLogger("../logs", name="NONE"),
    )