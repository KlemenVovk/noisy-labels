from pytorch_lightning.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from reproducibility.learning_strategies.ELRplus.utils import PreActResNet18
from noisypy.methods.learning_strategies.ELRplus.elr_plus import ELR_plus
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config
from noisypy.data.pipelines.index import AddIndex


class cifar10_clean_index_config(cifar10_base_config):

    dataset_train_augmentation = AddIndex()


class ELRplus_cifar10_clean_config(MethodConfig):

    data_config = cifar10_clean_index_config

    classifier = PreActResNet18
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = ELR_plus
    learning_strategy_args = dict(
        beta = 0.99, # β ∈ {0.5, 0.7, 0.9, 0.99}
        lmbd=1,      # λ ∈ {1, 3, 5, 7, 10}
        gamma=0.997, # γ ∈ [0, 1] 
        alpha=1,     # α ∈ {0, 0.1, 1, 2, 5}
        ema_update=True, # True or False
        ema_step=40000,  # EMA step (in iterations)
        coef_step=0,
    ) 

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=0.0005)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[150], gamma=0.1)

    trainer_args = dict(
        max_epochs=200 * 2,
        deterministic=True,
        logger=CSVLogger("../logs", name="ELRplus_cifar10_clean")
    )

    seed = 1337