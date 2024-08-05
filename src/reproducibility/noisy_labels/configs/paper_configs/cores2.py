from noisypy.methods.learning_strategies.cores2.cores2 import SampleSieve

from .base import BasePaperMethod


class cores2_config(BasePaperMethod):
    learning_strategy_cls = SampleSieve