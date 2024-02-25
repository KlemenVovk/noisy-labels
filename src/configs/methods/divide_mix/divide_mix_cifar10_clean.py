from pytorch_lightning.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from methods.learning_strategies.divide_mix.models import ResNet18
from methods.learning_strategies.divide_mix.divide_mix import DivideMix

from data.pipelines.divide_mix import DivideMixify
from configs.base import MethodConfig
from configs.data.cifar10 import cifar10_base_config, cifar10_train_transform


class divide_mix_cifar10_base_config(cifar10_base_config):
    num_train_samples = 5
    dataset_train_args = [
        dict(root="../data/cifar", mode="all", transform=cifar10_train_transform),
        dict(root="../data/cifar", mode="all", transform=cifar10_train_transform),
        dict(root="../data/cifar", mode="unlabeled", transform=cifar10_train_transform),
        dict(root="../data/cifar", mode="unlabeled", transform=cifar10_train_transform),
        dict(root="../data/cifar", mode="all", transform=cifar10_train_transform),
    ]

    dataset_train_augmentation = DivideMixify()

    
class divide_mix_cifar10_clean(MethodConfig):

    data_config = divide_mix_cifar10_base_config

    classifier = ResNet18
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = DivideMix
    learning_strategy_args = dict(warmup_epochs=10, 
                                  noise_type = "clean", 
                                  noise_rate = 0,
                                  p_thresh = 0.5, 
                                  temperature = 0.5, 
                                  alpha = 4)


    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = LambdaLR
    scheduler_args = dict(lr_lambda = lambda epoch: 0.1 if epoch >= 150 else 1)

    trainer_args = dict(
        max_epochs=100,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="divide_mix_clean")
    )

    seed = 1337