from noisypy.methods.learning_strategies.PESsemi.pes_semi import PES_semi
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..base.config import CIFAR100NoisyLabelsMethod


class PES_semi_config(CIFAR100NoisyLabelsMethod):

    learning_strategy_cls = PES_semi
    learning_strategy_args = dict(
        PES_lr=1e-4,
        warmup_epochs=35,
        T2=5,
        lambda_u=75,
        temperature=0.5,
        alpha=4,
        optimizer_refine_cls=Adam,
        model_type="paper_resnet",
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=0.0005)
    scheduler_cls = CosineAnnealingLR
    scheduler_args = dict(T_max=300, eta_min=0.02 / 100)

    trainer_args = {
        **CIFAR100NoisyLabelsMethod.trainer_args,
        "max_epochs": 300,
        "reload_dataloaders_every_n_epochs": 1,
    }
