from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from noisypy.methods.learning_strategies.peer_loss.utils import lr_plan
from noisypy.methods.learning_strategies.peer_loss.peer_loss import PeerLoss

from ..base import BenchmarkConfigCIFAR10N

stages = [0, 50]


class peer_loss_config(BenchmarkConfigCIFAR10N):
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
        **BenchmarkConfigCIFAR10N.trainer_args,
        "max_epochs": sum(stages) + 1,
    }
