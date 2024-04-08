from noisypy.methods.learning_strategies.volminnet.volminnet import VolMinNet
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from .base.config import NoisyLabelsMethod


class volminnet_config(NoisyLabelsMethod):

    learning_strategy_cls = VolMinNet
    learning_strategy_args = dict(
        lam=1e-4
    )

    optimizer_cls = [SGD, SGD]
    optimizer_args = [
        dict(lr=0.01, momentum=0.9, weight_decay=1e-4),
        dict(lr=0.01, momentum=0.9)
    ]
    scheduler_cls = [MultiStepLR, MultiStepLR]
    scheduler_args = [
        dict(milestones=[30, 60], gamma=0.1),
        dict(milestones=[30, 60], gamma=0.1),
    ]

    trainer_args = {
        **NoisyLabelsMethod.trainer_args,
        "max_epochs": 80,
    }
