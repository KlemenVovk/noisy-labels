from aim.pytorch_lightning import AimLogger

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import ToTensor

from methods.learning_strategies.co_teaching.co_teaching import CoTeaching
from methods.learning_strategies.co_teaching.utils import CNN, alpha_schedule

from configs.base import MethodConfig
from configs.data.cifar10 import cifar10_base_config
from data.pipelines.base import Compose
from data.pipelines.index import AddIndex
from data.pipelines.noise.noises import SymmetricNoise, InstanceNoise
from data.pipelines.noise.pipeline import AddNoise

# reproduces the original implementation, with their noise and all

class cifar10_noise_index(cifar10_base_config):

    dataset_train_augmentation = Compose([
        AddIndex(),
        AddNoise(SymmetricNoise(10, 0.2))
        #AddNoise(InstanceNoise(torch.load("configs/methods/co_teaching/reprod_assets/original_noise.pt"))),
    ])

    dataset_train_args = dict(
        transform=ToTensor()
    )
    dataset_val_args = dict(
        train=False,
        transform=ToTensor()
    )
    dataset_test_args = dict(
        train=False,
        transform=ToTensor()
    )
    datamodule_args = dict(
        batch_size=128,
        num_workers=0
    )

class co_teaching_reprod(MethodConfig):

    data_config = cifar10_noise_index

    classifier = CNN
    classifier_args = dict(
        input_channel=3,
        n_outputs=10
    )

    learning_strategy_cls = CoTeaching
    learning_strategy_args = dict(
        forget_rate=0.2,
        exponent=1,
        num_gradual=10,
        num_epochs=200,
    )

    optimizer_cls = Adam
    optimizer_args = dict(
        lr=0.001,
    )
    
    scheduler_cls = LambdaLR
    scheduler_args = dict(lr_lambda=alpha_schedule)

    trainer_args = dict(
        max_epochs=200,
        deterministic=True,
        logger=AimLogger(experiment="CoTeaching")
    )

    seed = 1