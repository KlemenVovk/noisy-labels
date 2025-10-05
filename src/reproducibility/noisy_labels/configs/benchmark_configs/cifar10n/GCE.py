from noisypy.methods.learning_strategies.GCE.GCE import GCE
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from ..base import BenchmarkConfigCIFAR10N
from ...base.wrappers import add_index_wrapper


class GCE_config(BenchmarkConfigCIFAR10N):
    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = GCE
    learning_strategy_args = dict(prune_start_epoch=40, prune_freq=10)

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[60], gamma=0.1)

    trainer_args = {
        **BenchmarkConfigCIFAR10N.trainer_args,
        "enable_checkpointing": True,
        "max_epochs": 100,
    }
