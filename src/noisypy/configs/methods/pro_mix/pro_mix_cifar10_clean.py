from pytorch_lightning.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.pro_mix.pro_mix import ProMix
from noisypy.data.pipelines.pro_mix import ProMixify
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config, cifar10_train_transform


class cifar10_clean_pro_mix_config(cifar10_base_config):

    dataset_train_args = [
        dict(mode="all", transform=cifar10_train_transform),
        dict(mode="all", transform=cifar10_train_transform),
    ]

    datamodule_args = dict(
        batch_size=256,
        num_workers=2
    )

    dataset_train_augmentation = ProMixify()

    
class pro_mix_cifar10_clean_config(MethodConfig):

    data_config = cifar10_clean_pro_mix_config

    classifier = resnet34
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = ProMix
    learning_strategy_args = dict(
        warmup_epochs=10, 
        rampup_epochs = 50,
        noise_type = "clean", 
        rho_start = 0.2,
        rho_end = 0.6,
        debias_beta_pl = 0.8,
        alpha_output = 0.8,
        tau = 0.99,
        start_expand = 250,
        threshold = 0.9,
        bias_m = 0.9999,
        temperature = 0.5, 
        model_type = "pytorch_resnet",
        feat_dim = 128,
    )


    optimizer_cls = SGD
    optimizer_args = dict(lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = CosineAnnealingLR
    scheduler_args = dict(T_max=600, eta_min=5e-5)

    trainer_args = dict(
        max_epochs=600,
        deterministic=True,
        logger=CSVLogger("../logs", name="pro_mix_cifar10_clean")
    )

    seed = 1337