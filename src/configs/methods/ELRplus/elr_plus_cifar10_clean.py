from pytorch_lightning.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from methods.learning_strategies.ELRplus.models import PreActResNet18
from methods.learning_strategies.ELRplus.elr_plus import ELR_plus

from configs.base import MethodConfig
from configs.methods.ELR.elr_cifar10_clean import cifar10_index_config


class elr_plus_cifar10_clean(MethodConfig):

    data_config = cifar10_index_config

    classifier = PreActResNet18
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = ELR_plus
    learning_strategy_args = dict(beta = 0.99, # β ∈ {0.5, 0.7, 0.9, 0.99}
                                  lmbd=1,      # λ ∈ {1, 3, 5, 7, 10}
                                  gamma=0.997, # γ ∈ [0, 1] 
                                  alpha=1,     # α ∈ {0, 0.1, 1, 2, 5}
                                  ema_update=True, # True or False
                                  ema_step=40000,  # EMA step (in iterations)
                                  coef_step=0 
    ) 

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=0.0005)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[150], gamma=0.1)

    trainer_args = dict(
        max_epochs=200 * 2,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="elr_plus_clean")
    )

    seed = 1337