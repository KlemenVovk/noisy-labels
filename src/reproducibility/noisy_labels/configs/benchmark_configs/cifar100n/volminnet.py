from noisypy.methods.learning_strategies.volminnet.volminnet import VolMinNet
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from ..base import BenchmarkConfigCIFAR100N


class volminnet_config(BenchmarkConfigCIFAR100N):

    learning_strategy_cls = VolMinNet
    learning_strategy_args = dict(lam=1e-4, init_t=4.5)

    optimizer_cls = [SGD, Adam]
    optimizer_args = [
        dict(lr=0.01, momentum=0.9, weight_decay=1e-4),
        dict(lr=0.01),
    ]
    scheduler_cls = [MultiStepLR, MultiStepLR]
    scheduler_args = [
        dict(milestones=[30, 60], gamma=0.1),
        dict(milestones=[30, 60], gamma=0.1),
    ]

    trainer_args = {
        **BenchmarkConfigCIFAR100N.trainer_args,
        "max_epochs": 80,
    }
