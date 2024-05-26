from noisypy.methods.learning_strategies.CE.CE import CE
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from .base.config import NoisyLabelsMethod


class CE_config(NoisyLabelsMethod):

    learning_strategy_cls = CE

    optimizer_cls=SGD
    optimizer_args=dict(
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler_cls=MultiStepLR
    scheduler_args=dict(
        milestones=[60],
        gamma=0.1
    )

    trainer_args = {
        **NoisyLabelsMethod.trainer_args,
        "max_epochs": 100,
    }