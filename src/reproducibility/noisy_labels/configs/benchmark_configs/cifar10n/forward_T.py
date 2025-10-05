from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from noisypy.methods.learning_strategies.FBT.FBT import ForwardT

from ..base import BenchmarkConfigCIFAR10N


class forwardT_config(BenchmarkConfigCIFAR10N):
    learning_strategy_cls = ForwardT
    learning_strategy_args = dict(
        warmup_epochs=120,
        filter_outliers=False,
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[40, 80], gamma=0.1)

    trainer_args = {
        **BenchmarkConfigCIFAR10N.trainer_args,
        "max_epochs": 240,
        "num_sanity_val_steps": 0,
    }
