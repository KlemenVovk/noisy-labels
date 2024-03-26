from noisypy.methods.learning_strategies.PESsemi.pes_semi import PES_semi
from .base.config import NoisyLabelsMethod
from torch.optim import Adam


class PES_semi_config(NoisyLabelsMethod):

    learning_strategy_cls = PES_semi
    learning_strategy_args = dict(
        PES_lr=1e-4,
        T1=20,
        T2=5,
        lambda_u = 5,
        temperature = 0.5,
        alpha = 4,
        optimizer_refine_cls=Adam,
    )
