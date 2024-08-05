from noisypy.methods.learning_strategies.ELRplus.elr_plus import ELR_plus
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from ..base.utils import PreResNet18
from ..base.config import CIFAR100NoisyLabelsMethod
from ..base.wrappers import add_index_wrapper


class ELR_plus_config(CIFAR100NoisyLabelsMethod):
    classifier = PreResNet18

    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = ELR_plus
    learning_strategy_args = dict(
        beta=0.9,  # β ∈ {0.5, 0.7, 0.9, 0.99}
        lmbd=7,  # λ ∈ {1, 3, 5, 7, 10}
        gamma=0.997,  # γ ∈ [0, 1]
        alpha=1,  # α ∈ {0, 0.1, 1, 2, 5}
        ema_update=True,  # True or False
        ema_step=40000,  # EMA step (in iterations)
        coef_step=40000,
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=0.0005)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[200], gamma=0.1)

    trainer_args = {
        **CIFAR100NoisyLabelsMethod.trainer_args,
        "max_epochs": 250 * 2,
    }
