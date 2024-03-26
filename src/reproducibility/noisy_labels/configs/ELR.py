from noisypy.methods.learning_strategies.ELR.elr import ELR
from .base.config import NoisyLabelsMethod
from .base.wrappers import add_index_wrapper


class ELR_config(NoisyLabelsMethod):

    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = ELR
    learning_strategy_args = dict(
        beta=0.99,
        lmbd=1
    )