from noisypy.methods.learning_strategies.jocor.jocor import JoCoR
from noisypy.methods.learning_strategies.co_teaching.utils import alpha_schedule
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from ..base import BenchmarkConfigCIFAR10N


class jocor_config(BenchmarkConfigCIFAR10N):
    learning_strategy_cls = JoCoR
    learning_strategy_args = dict(
        forget_rate=0.2,
        exponent=1,
        num_gradual=10,
        co_lambda=0.9,
        num_epochs=200,
    )

    optimizer_cls = Adam
    optimizer_args = dict(lr=0.001)
    scheduler_cls = LambdaLR
    scheduler_args = dict(lr_lambda=alpha_schedule)

    trainer_args = {
        **BenchmarkConfigCIFAR10N.trainer_args,
        "max_epochs": 200,
    }
