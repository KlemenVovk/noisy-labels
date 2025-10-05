from noisypy.methods.learning_strategies.GCE.GCE import GCE

from ..base.wrappers import add_index_wrapper
from .base import BasePaperMethod


class GCE_config(BasePaperMethod):
    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = GCE
    learning_strategy_args = dict(prune_start_epoch=40, prune_freq=10)
