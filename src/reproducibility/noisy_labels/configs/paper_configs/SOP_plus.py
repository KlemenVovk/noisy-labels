from noisypy.methods.learning_strategies.SOPplus.sop_plus import SOPplus
from torch.optim import SGD

from ..base.utils import PreResNet18
from ..base.wrappers import double_aug_wrapper
from .base import BasePaperMethod


class SOP_plus_config(BasePaperMethod):
    _data_config_wrapper = double_aug_wrapper

    # https://openreview.net/pdf?id=TBWA6PLJZQm special treatment for SOP the pre-act resnet is used, but other parameters are default
    classifier = PreResNet18

    learning_strategy_cls = SOPplus
    learning_strategy_args = dict(
        ratio_consistency=0.9,
        ratio_balance=0.1,
        lr_u=10,
        lr_v=10,
        overparam_mean=0.0,
        overparam_std=1e-8,
        overparam_momentum=0,
        overparam_weight_decay=0,
        overparam_optimizer_cls=SGD,
    )
