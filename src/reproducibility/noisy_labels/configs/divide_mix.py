from noisypy.methods.learning_strategies.divide_mix.divide_mix import DivideMix
from .base.config import NoisyLabelsMethod
from .base.wrappers import dividemixify_wrapper


class divide_mix_config(NoisyLabelsMethod):

    _data_config_wrapper = dividemixify_wrapper

    learning_strategy_cls = DivideMix
    learning_strategy_args = dict(
        warmup_epochs=10, 
        noise_type = "clean", 
        noise_rate = 0,
        p_thresh = 0.5, 
        temperature = 0.5, 
        alpha = 4
    )