import torch
from pytorch_lightning.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

from noisypy.methods.learning_strategies.DISC.DISC import DISC
from noisypy.data.pipelines.base import Compose
from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.noise.noises import InstanceNoise
from noisypy.data.pipelines.index import AddIndex
from noisypy.data.pipelines.double_augmentation import DoubleAugmentation
from noisypy.configs.base.data import DataConfig
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import CIFAR10, cifar10_train_transform, cifar10_test_transform

from .utils import resnet18


weak_train_transform = cifar10_train_transform
strong_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class cifar10_DISC_repro(DataConfig):
   
    dataset_args = dict(
        root="../data/cifar",
        download=True
    )
    
    dataset_train_cls = CIFAR10
    dataset_train_args = dict(
        transform=None
    )
    dataset_train_augmentation = Compose([
        AddNoise(InstanceNoise(torch.load("src/reproducibility/learning_strategies/DISC/assets/original_sym20_noise.pt", weights_only=False))),
        AddIndex(),
        DoubleAugmentation(weak_train_transform, strong_train_transform),
    ])
    
    dataset_val_cls = CIFAR10
    dataset_val_args = dict(
        train=False,
        transform=cifar10_test_transform
    )
    
    dataset_test_cls = CIFAR10
    dataset_test_args = dict(
        train=False,
        transform=cifar10_test_transform
    )

    datamodule_args = dict(
        batch_size=128,
        num_workers=4
    )


    
class DISC_cifar10_repro_config(MethodConfig):

    data_config = cifar10_DISC_repro

    classifier = resnet18
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = DISC
    learning_strategy_args = dict(
        start_epoch=20,
        alpha=5.0, 
        sigma=0.5,
        momentum=0.99,
        lambd_ce=1.0,
        lambd_h=1.0,
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0, weight_decay=0.001)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[80, 160])

    trainer_args = dict(
        max_epochs=200,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        logger=CSVLogger("../logs", name="disc_cifar10_repro")
    )

    seed = 1
