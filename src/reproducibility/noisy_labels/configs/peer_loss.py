from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from noisypy.methods.learning_strategies.peer_loss.utils import lr_plan
from noisypy.methods.learning_strategies.peer_loss.peer_loss import PeerLoss

from .base.utils import PreResNet18
from .base.config import NoisyLabelsMethod

stages = [0, 50]


class peer_loss_config(NoisyLabelsMethod):
    classifier = PreResNet18

    learning_strategy_cls = PeerLoss
    learning_strategy_args = dict(stage_epochs=stages)

    optimizer_cls = [SGD, SGD]
    optimizer_args = [
        dict(lr=1, weight_decay=1e-4),
        dict(lr=1, weight_decay=1e-4),
    ]

    scheduler_cls = [LambdaLR, LambdaLR]
    scheduler_args = [
        dict(lr_lambda=lambda e: lr_plan(e)),
        dict(lr_lambda=lambda e: lr_plan(e + stages[0])),
    ]

    trainer_args = {
        **NoisyLabelsMethod.trainer_args,
        "max_epochs": sum(stages) + 1,
    }
