from noisypy.methods.learning_strategies.CAL.cal import CAL
from noisypy.methods.learning_strategies.CAL.utils import SegAlpha

from .base import BasePaperMethod
from ..base.wrappers import add_index_wrapper


class CAL_config(BasePaperMethod):
    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = CAL
    learning_strategy_args = dict(
        alpha=0.0,
        warmup_epochs=100,
        alpha_scheduler_cls=SegAlpha,
        alpha_scheduler_args=dict(
            alpha_list=[0.0, 1.0, 1.0],
            milestones=[10, 40, 80],
        ),
        alpha_scheduler_args_warmup=dict(
            alpha_list=[0.0, 2.0, 2.0],
            milestones=[10, 40, 80],
        ),
    )

    trainer_args = {
        **BasePaperMethod.trainer_args,
        "max_epochs": 200,
    }
