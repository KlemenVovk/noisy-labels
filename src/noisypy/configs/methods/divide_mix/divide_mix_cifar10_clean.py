from pytorch_lightning.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.divide_mix.divide_mix import DivideMix
from noisypy.data.pipelines.divide_mix import DivideMixify
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config, cifar10_train_transform, cifar10_test_transform


class cifar10_clean_divide_mix_config(cifar10_base_config):

    dataset_train_args = [
        dict(mode="all",        transform=cifar10_train_transform),
        dict(mode="all",        transform=cifar10_train_transform),
        dict(mode="unlabeled",  transform=cifar10_train_transform),
        dict(mode="unlabeled",  transform=cifar10_train_transform),
        dict(mode="all",        transform=cifar10_test_transform),
    ]

    dataset_train_augmentation = DivideMixify()

    
class divide_mix_cifar10_clean_config(MethodConfig):

    data_config = cifar10_clean_divide_mix_config

    classifier = resnet34
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = DivideMix
    learning_strategy_args = dict(
        warmup_epochs=10, 
        noise_type = "clean", 
        noise_rate = 0,
        p_thresh = 0.5, 
        temperature = 0.5, 
        alpha = 4
    )


    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = LambdaLR
    scheduler_args = dict(lr_lambda = lambda epoch: 0.1 if epoch >= 150 else 1)

    trainer_args = dict(
        max_epochs=100,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        logger=CSVLogger("../logs", name="divide_mix_cifar10_clean")
    )

    seed = 1337