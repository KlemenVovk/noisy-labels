from noisypy.methods.learning_strategies.CE.CE import CE
from .base.config import NoisyLabelsMethod


class CE_config(NoisyLabelsMethod):

    learning_strategy_cls = CE