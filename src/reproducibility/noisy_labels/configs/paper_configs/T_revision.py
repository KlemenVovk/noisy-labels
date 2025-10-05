from noisypy.methods.learning_strategies.t_revision.t_revision import TRevision

from .base import BasePaperMethod

# First stage is warmup, as per https://openreview.net/pdf?id=TBWA6PLJZQm E.3
# and communication with authors using 200 epochs with CE.
stages = [200, 30, 30]


class TRevision_config(BasePaperMethod):
    learning_strategy_cls = TRevision
    learning_strategy_args = dict(stage_epochs=stages)

    optimizer_cls = [BasePaperMethod.optimizer_cls] * 3
    optimizer_args = [BasePaperMethod.optimizer_args] * 3

    scheduler_cls = [BasePaperMethod.scheduler_cls] * 3
    scheduler_args = [BasePaperMethod.scheduler_args] * 3

    trainer_args = {
        **BasePaperMethod.trainer_args,
        "max_epochs": sum(stages) + 1,
        "num_sanity_val_steps": 0,
    }
