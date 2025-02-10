from noisypy.methods.learning_strategies.volminnet.volminnet import VolMinNet
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from ..base import BenchmarkConfigCIFAR10N


class volminnet_config(BenchmarkConfigCIFAR10N):

    learning_strategy_cls = VolMinNet
    learning_strategy_args = dict(lam=1e-4, init_t=2)

    optimizer_cls = [SGD, SGD]
    optimizer_args = [
        dict(lr=0.01, momentum=0.9, weight_decay=1e-4),
        dict(lr=0.01, momentum=0.9),
    ]
    scheduler_cls = [MultiStepLR, MultiStepLR]
    scheduler_args = [
        dict(milestones=[30, 60], gamma=0.1),
        dict(milestones=[30, 60], gamma=0.1),
    ]

    trainer_args = {
        **BenchmarkConfigCIFAR10N.trainer_args,
        "max_epochs": 80,
    }
