from noisypy.methods.learning_strategies.SOP.sop import SOP
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from ..base import BenchmarkConfigCIFAR10N
from ...base.wrappers import add_index_wrapper


class SOP_config(BenchmarkConfigCIFAR10N):

    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = SOP
    learning_strategy_args = dict(
        ratio_consistency=0,
        ratio_balance=0,
        lr_u=10,
        lr_v=10,
        overparam_mean=0.0,
        overparam_std=1e-8,
        overparam_momentum=0,
        overparam_weight_decay=0,
        overparam_optimizer_cls=SGD,
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[40, 80], gamma=0.1)

    trainer_args = {
        **BenchmarkConfigCIFAR10N.trainer_args,
        "max_epochs": 120,
    }
