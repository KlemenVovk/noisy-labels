from noisypy.methods.learning_strategies.jocor.jocor import JoCoR
from .base.config import NoisyLabelsMethod


class jocor_config(NoisyLabelsMethod):

    learning_strategy_cls = JoCoR
    learning_strategy_args = dict(
        forget_rate=0.2,
        exponent=1,
        num_gradual=10,
        co_lambda=0.9,
        num_epochs=NoisyLabelsMethod.trainer_args["max_epochs"],
    )