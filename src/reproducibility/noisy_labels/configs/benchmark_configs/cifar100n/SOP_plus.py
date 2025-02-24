from noisypy.methods.learning_strategies.SOPplus.sop_plus import SOPplus
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..base import BenchmarkConfigCIFAR100N
from ...base.wrappers import double_aug_wrapper


class SOP_plus_config(BenchmarkConfigCIFAR100N):
    _data_config_wrapper = double_aug_wrapper

    learning_strategy_cls = SOPplus
    learning_strategy_args = dict(
        ratio_consistency=0.9,
        ratio_balance=0.1,
        lr_u=1,
        lr_v=10,
        overparam_mean=0.0,
        overparam_std=1e-8,
        overparam_momentum=0,
        overparam_weight_decay=0,
        overparam_optimizer_cls=SGD,
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = CosineAnnealingLR
    scheduler_args = dict(T_max=300, eta_min=0.0002)

    trainer_args = {
        **BenchmarkConfigCIFAR100N.trainer_args,
        "max_epochs": 300,
    }
