from pytorch_lightning.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from methods.learning_strategies.ELR.models import resnet34
from methods.learning_strategies.ELR.elr import ELR

from data.pipelines.index import AddIndex
from configs.base import MethodConfig
from configs.data.cifar10 import cifar10_base_config

class cifar10_index_config(cifar10_base_config):

    dataset_train_augmentation = AddIndex()    

class elr_cifar10_clean(MethodConfig):

    data_config = cifar10_index_config

    classifier = resnet34
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = ELR
    learning_strategy_args = dict(beta = 0.99, lmbd=1)  # β ∈ {0.5, 0.7, 0.9, 0.99}, λ ∈ {1, 3, 5, 7, 10}

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=0.001)
    scheduler_cls = CosineAnnealingWarmRestarts
    scheduler_args = dict(T_0=10, eta_min=0.001)

    trainer_args = dict(
        max_epochs=120,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="elr_clean")
    )

    seed = 1337