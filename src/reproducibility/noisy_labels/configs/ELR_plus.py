from noisypy.methods.learning_strategies.ELRplus.elr_plus import ELR_plus
from .base.config import NoisyLabelsMethod
from .base.wrappers import add_index_wrapper


class ELR_plus_config(NoisyLabelsMethod):

    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = ELR_plus
    learning_strategy_args = dict(
        beta = 0.99, # β ∈ {0.5, 0.7, 0.9, 0.99}
        lmbd=1,      # λ ∈ {1, 3, 5, 7, 10}
        gamma=0.997, # γ ∈ [0, 1] 
        alpha=1,     # α ∈ {0, 0.1, 1, 2, 5}
        ema_update=True, # True or False
        ema_step=40000,  # EMA step (in iterations)
        coef_step=0,
    )