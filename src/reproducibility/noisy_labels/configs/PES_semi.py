from noisypy.methods.learning_strategies.PESsemi.pes_semi import PES_semi
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from .base.config import NoisyLabelsMethod, CSVLogger


class PES_semi_config(NoisyLabelsMethod):

    learning_strategy_cls = PES_semi
    learning_strategy_args = dict(
        PES_lr=1e-4,
        warmup_epochs=20,
        T2=5,
        lambda_u = 5,
        temperature = 0.5,
        alpha = 4,
        optimizer_refine_cls=Adam,
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=0.0005)
    scheduler_cls = CosineAnnealingLR
    scheduler_args = dict(T_max=300, eta_min = 0.1 / 100)

    trainer_args = dict(
        reload_dataloaders_every_n_epochs=1,
        max_epochs=100,
        deterministic=True,
        logger=CSVLogger("../logs", name="NONE"),
    )