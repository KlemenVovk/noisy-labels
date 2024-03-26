from noisypy.methods.learning_strategies.CAL.cal import CAL
from noisypy.methods.learning_strategies.CAL.utils import SegAlpha
from .base.config import NoisyLabelsMethod
from .base.wrappers import add_index_wrapper


class CAL_config(NoisyLabelsMethod):

    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = CAL
    learning_strategy_args = dict(
        alpha = 0.0, 
        warmup_epochs = 65,
        alpha_scheduler_cls = SegAlpha,
        alpha_scheduler_args = dict(
            alpha_list = [0.0, 1.0, 1.0],
            milestones = [10, 40, 80],
        ),
        alpha_scheduler_args_warmup = dict(
            alpha_list = [0.0, 2.0],
            milestones = [10, 40],
        ),
    )