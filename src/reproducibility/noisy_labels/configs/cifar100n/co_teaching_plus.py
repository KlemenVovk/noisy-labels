from functools import partial
from noisypy.methods.learning_strategies.co_teaching.co_teaching import CoTeachingPlus
from noisypy.methods.learning_strategies.co_teaching.utils import alpha_schedule
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from ..base.config import CIFAR100NoisyLabelsMethod
from ..base.wrappers import add_index_wrapper


class co_teaching_plus_config(CIFAR100NoisyLabelsMethod):

    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = CoTeachingPlus
    learning_strategy_args = dict(
        forget_rate=0.2,
        exponent=1,
        num_gradual=10,
        num_epochs=200,
        init_epoch=20,
    )

    optimizer_cls = Adam
    optimizer_args = dict(
        lr=0.001,
    )

    alpha_schedule = partial(alpha_schedule, decay_start_epoch=100)
    scheduler_cls = LambdaLR
    scheduler_args = dict(lr_lambda=alpha_schedule)

    trainer_args = {
        **CIFAR100NoisyLabelsMethod.trainer_args,
        "max_epochs": 200,
    }
