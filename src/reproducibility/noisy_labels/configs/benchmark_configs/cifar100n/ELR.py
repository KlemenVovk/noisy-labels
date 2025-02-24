from noisypy.methods.learning_strategies.ELR.elr import ELR
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from ..base import BenchmarkConfigCIFAR100N
from ...base.wrappers import add_index_wrapper


class ELR_config(BenchmarkConfigCIFAR100N):
    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = ELR
    learning_strategy_args = dict(beta=0.9, lmbd=7)

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=0.001)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[80, 120], gamma=0.01)

    trainer_args = {
        **BenchmarkConfigCIFAR100N.trainer_args,
        "max_epochs": 150,
    }
