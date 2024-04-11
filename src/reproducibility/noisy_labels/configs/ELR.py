from noisypy.methods.learning_strategies.ELR.elr import ELR
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .base.config import NoisyLabelsMethod
from .base.wrappers import add_index_wrapper


class ELR_config(NoisyLabelsMethod):

    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = ELR
    learning_strategy_args = dict(
        beta=0.7,
        lmbd=3
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=0.001)
    scheduler_cls = CosineAnnealingWarmRestarts
    scheduler_args = dict(T_0=10, eta_min=0.001)

    trainer_args = {
        **NoisyLabelsMethod.trainer_args,
        "max_epochs": 120,
    }
