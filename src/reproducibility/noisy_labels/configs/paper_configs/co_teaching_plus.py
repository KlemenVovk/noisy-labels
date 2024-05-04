from noisypy.methods.learning_strategies.co_teaching.co_teaching import CoTeachingPlus
from noisypy.methods.learning_strategies.co_teaching.utils import alpha_schedule
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from ..base.wrappers import add_index_wrapper
from .base import BasePaperMethod


class co_teaching_plus_config(BasePaperMethod):

    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = CoTeachingPlus
    learning_strategy_args = dict(
        forget_rate=0.2,
        exponent=1,
        num_gradual=10,
        num_epochs=100, # this was 200
        init_epoch=20,
    )
