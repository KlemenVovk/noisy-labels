from noisypy.methods.learning_strategies.PES.pes import PES
from .base.config import NoisyLabelsMethod
from torch.optim import Adam


class PES_config(NoisyLabelsMethod):

    learning_strategy_cls = PES
    learning_strategy_args = dict(
        PES_lr=1e-4,
        T1=25,
        T2=7,
        T3=5,
        optimizer_refine_cls=Adam,
    )
