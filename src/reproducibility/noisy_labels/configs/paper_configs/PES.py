from noisypy.methods.learning_strategies.PES.pes import PES
from torch.optim import Adam

from .base import BasePaperMethod


class PES_config(BasePaperMethod):
    learning_strategy_cls = PES
    learning_strategy_args = dict(
        PES_lr=1e-4,
        warmup_epochs=25,
        T2=7,
        T3=5,
        optimizer_refine_cls=Adam,
        model_type="paper_resnet",
    )
