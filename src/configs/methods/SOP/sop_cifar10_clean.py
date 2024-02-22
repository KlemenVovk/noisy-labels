from pytorch_lightning.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from methods.learning_strategies.SOP.utils import resnet34
from methods.learning_strategies.SOP.sop import SOP

from data.pipelines.base import Compose
from data.pipelines.index import AddIndex
from data.pipelines.two_images import TwoImages
from configs.base import MethodConfig
from configs.data.cifar10 import cifar10_base_config

class cifar10_two_images_index_config(cifar10_base_config):

    dataset_train_augmentation = Compose([AddIndex(), TwoImages()])

class sop_cifar10_clean(MethodConfig):

    data_config = cifar10_two_images_index_config

    classifier = resnet34
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = SOP
    learning_strategy_args = dict(
        ratio_consistency = 0,
        ratio_balance = 0,
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
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[40, 80], gamma=0.1)

    trainer_args = dict(
        max_epochs=120,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="sop_clean")
    )

    seed = 1337