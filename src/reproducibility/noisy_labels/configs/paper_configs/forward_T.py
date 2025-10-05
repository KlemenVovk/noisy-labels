from noisypy.methods.learning_strategies.FBT.FBT import ForwardT

from .base import BasePaperMethod


class forwardT_config(BasePaperMethod):
    learning_strategy_cls = ForwardT
    learning_strategy_args = dict(
        warmup_epochs=200,  # See https://openreview.net/pdf?id=TBWA6PLJZQm E.3 + communication with authors.
        filter_outliers=False,
    )

    trainer_args = {
        **BasePaperMethod.trainer_args,
        "max_epochs": 300,  # 200 warmup + 100 training
        "num_sanity_val_steps": 0,
    }
