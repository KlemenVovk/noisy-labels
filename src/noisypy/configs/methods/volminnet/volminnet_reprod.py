from aim.pytorch_lightning import AimLogger
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from noisypy.methods.learning_strategies.volminnet.volminnet import VolMinNet
from noisypy.methods.learning_strategies.volminnet.utils import ResNet18
from noisypy.data.pipelines.noise.noises import SymmetricNoise
from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config


class cifar10_noise(cifar10_base_config):

    dataset_train_augmentation = AddNoise(SymmetricNoise(10, 0.2))


class volminnet_reprod(MethodConfig):

    data_config = cifar10_noise

    classifier = ResNet18
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = VolMinNet
    learning_strategy_args = dict(
        lam=1e-4
    )

    optimizer_cls = [SGD, SGD]
    optimizer_args = [
        dict(lr=0.01, momentum=0.9, weight_decay=1e-4),
        dict(lr=0.01, momentum=0.9)
    ]
    
    scheduler_cls = [MultiStepLR, MultiStepLR]
    scheduler_args = [
        dict(milestones=[30, 60], gamma=0.1),
        dict(milestones=[30, 60], gamma=0.1),
    ]

    trainer_args = dict(
        max_epochs=80,
        deterministic=True,
        logger=AimLogger(experiment="VolMinNet")
    )

    seed = 1337