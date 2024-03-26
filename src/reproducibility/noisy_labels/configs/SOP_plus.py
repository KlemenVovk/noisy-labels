from noisypy.methods.learning_strategies.SOPplus.sop_plus import SOPplus
from .base.config import NoisyLabelsMethod
from .base.wrappers import double_aug_wrapper
from torch.optim import SGD


class SOP_plus_config(NoisyLabelsMethod):

    _data_config_wrapper = double_aug_wrapper

    learning_strategy_cls = SOPplus
    learning_strategy_args = dict(
        ratio_consistency = 0.9,
        ratio_balance = 0.1,
        lr_u = 10,
        lr_v = 10,
        overparam_mean = 0.0,
        overparam_std = 1e-8,
        overparam_momentum = 0,
        overparam_weight_decay = 0,
        overparam_optimizer_cls = SGD
    )
