from noisypy.methods.learning_strategies.co_teaching.co_teaching import CoTeachingPlus
from .base.config import NoisyLabelsMethod
from .base.wrappers import add_index_wrapper


class co_teaching_plus_config(NoisyLabelsMethod):

    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = CoTeachingPlus
    learning_strategy_args = dict(
        forget_rate=0.2,
        exponent=1,
        num_gradual=10,
        num_epochs=200,
        init_epoch=20,
    )