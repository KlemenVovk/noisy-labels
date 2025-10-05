from noisypy.methods.learning_strategies.volminnet.volminnet import VolMinNet
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from .base import BasePaperMethod


class volminnet_config(BasePaperMethod):
    learning_strategy_cls = VolMinNet
    learning_strategy_args = dict(lam=1e-4, init_t=2)

    # two copies of the original optimizer and scheduler
    optimizer_cls = [SGD, SGD]
    optimizer_args = [
        dict(lr=0.1, momentum=0.9, weight_decay=5e-4),
        dict(lr=0.1, momentum=0.9, weight_decay=5e-4),
    ]
    scheduler_cls = [MultiStepLR, MultiStepLR]
    scheduler_args = [
        dict(milestones=[60], gamma=0.1),
        dict(milestones=[60], gamma=0.1),
    ]
