from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from noisypy.methods.learning_strategies.t_revision.t_revision import TRevision

from .base.utils import ResNet18
from .base.config import NoisyLabelsMethod


stages = [20, 30, 30]


class TRevision_config(NoisyLabelsMethod):
    classifier = ResNet18

    learning_strategy_cls = TRevision
    learning_strategy_args = dict(stage_epochs=stages)

    optimizer_cls = [SGD, SGD, Adam]
    optimizer_args = [
        dict(lr=0.01, weight_decay=1e-4),
        dict(lr=0.01, weight_decay=1e-4, momentum=0.9),
        dict(lr=5e-7, weight_decay=1e-4),
    ]

    scheduler_cls = [LambdaLR, MultiStepLR, LambdaLR]
    scheduler_args = [
        dict(lr_lambda=lambda _: 1),
        dict(milestones=[40, 80], gamma=0.1),
        dict(lr_lambda=lambda _: 1),
    ]

    trainer_args = {
        **NoisyLabelsMethod.trainer_args,
        "max_epochs": sum(stages) + 1,
        "num_sanity_val_steps": 0,
    }
