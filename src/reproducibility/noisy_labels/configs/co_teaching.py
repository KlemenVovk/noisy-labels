from noisypy.methods.learning_strategies.co_teaching.co_teaching import CoTeaching
from noisypy.methods.learning_strategies.co_teaching.utils import alpha_schedule
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from .base.config import NoisyLabelsMethod
from .base.wrappers import add_index_wrapper


class co_teaching_config(NoisyLabelsMethod):
    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = CoTeaching
    learning_strategy_args = dict(
        forget_rate=0.2,
        exponent=1,
        num_gradual=10,
        num_epochs=200,
    )

    optimizer_cls = Adam
    optimizer_args = dict(
        lr=0.001,
    )
    scheduler_cls = LambdaLR
    scheduler_args = dict(lr_lambda=alpha_schedule)

    trainer_args = {
        **NoisyLabelsMethod.trainer_args,
        "max_epochs": 200,
    }
