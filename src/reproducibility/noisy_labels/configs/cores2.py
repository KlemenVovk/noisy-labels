from noisypy.methods.learning_strategies.cores2.cores2 import SampleSieve
from .base.config import NoisyLabelsMethod


class cores2_config(NoisyLabelsMethod):

    learning_strategy_cls = SampleSieve