from lightning import Trainer
from aim.pytorch_lightning import AimLogger

from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from methods.learning_strategies.t_revision.utils import ResNet18
from methods.learning_strategies.t_revision.t_revision import TRevision

from data.pipelines.noise.noises import SymmetricNoise
from data.pipelines.noise.pipeline import AddNoise

from configs.base import MethodConfig
from configs.data.cifar10 import cifar10_base_config


class cifar10_noise(cifar10_base_config):

    dataset_train_augmentation = AddNoise(SymmetricNoise(10, 0.2))
    

stages = [20, 30, 30]

class t_revision_reprod(MethodConfig):

    data_config = cifar10_noise

    classifier = ResNet18
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = TRevision
    learning_strategy_args = dict(
        stage_epochs=stages
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[40, 80], gamma=0.1)

    trainer_args = dict(
        max_epochs=sum(stages)+1,
        deterministic=True,
        logger=AimLogger(experiment="T-Revision")
    )

    seed = 1337