from noisypy.methods.learning_strategies.PESsemi.pes_semi import PES_semi
from torch.optim import Adam

from .base import BasePaperMethod


class PES_semi_config(BasePaperMethod):

    learning_strategy_cls = PES_semi
    learning_strategy_args = dict(
        PES_lr=1e-4,
        warmup_epochs=20,
        T2=5,
        lambda_u = 5,
        temperature = 0.5,
        alpha = 4,
        optimizer_refine_cls=Adam,
        model_type='paper_resnet',
    )

    trainer_args = {
        **BasePaperMethod.trainer_args,
        "reload_dataloaders_every_n_epochs": 1,
    }