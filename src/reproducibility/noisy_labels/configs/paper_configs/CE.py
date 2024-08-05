from noisypy.methods.learning_strategies.CE.CE import CE

from .base import BasePaperMethod


class CE_config(BasePaperMethod):
    learning_strategy_cls = CE
