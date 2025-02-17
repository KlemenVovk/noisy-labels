from noisypy.methods.learning_strategies.FBT.FBT import BackwardT

from .base import BasePaperMethod


class backwardT_config(BasePaperMethod):

    learning_strategy_cls = BackwardT
    learning_strategy_args = dict(
        warmup_epochs=0,
        filter_outliers=False,
    )

    trainer_args = {
        **BasePaperMethod.trainer_args,
        "num_sanity_val_steps": 0,
    }
