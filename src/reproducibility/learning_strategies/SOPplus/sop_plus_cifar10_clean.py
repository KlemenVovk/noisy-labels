from pytorch_lightning.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..SOP.utils import PreActResNet18
from noisypy.methods.learning_strategies.SOPplus.sop_plus import SOPplus
from noisypy.methods.learning_strategies.SOPplus.utils import autoaug_paper_cifar10
from noisypy.data.pipelines.base import Compose
from noisypy.data.pipelines.index import AddIndex
from noisypy.data.pipelines.double_augmentation import DoubleAugmentation
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config, cifar10_train_transform


transform1 = cifar10_train_transform
transform2 = autoaug_paper_cifar10

class cifar10_double_augmentation_index_config(cifar10_base_config):
    dataset_train_args = dict(
        transform=None
    )

    dataset_train_augmentation = Compose([AddIndex(), DoubleAugmentation(transform1=transform1, transform2=transform2)])


class sop_plus_cifar10_clean(MethodConfig):

    data_config = cifar10_double_augmentation_index_config

    classifier = PreActResNet18
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = SOPplus
    learning_strategy_args = dict(
        ratio_consistency = 0.9,
        ratio_balance = 0.1,
        lr_u = 10,
        lr_v = 10,
        overparam_mean = 0.0,
        overparam_std = 1e-8,
        overparam_momentum = 0,
        overparam_weight_decay = 0,
        overparam_optimizer_cls = SGD
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = CosineAnnealingLR
    scheduler_args = dict(T_max=300, eta_min=0.0002)

    trainer_args = dict(
        max_epochs=300,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="sop_plus_clean")
    )

    seed = 1337