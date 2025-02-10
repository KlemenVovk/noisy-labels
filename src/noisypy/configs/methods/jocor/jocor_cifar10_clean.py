from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch.loggers import CSVLogger

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.jocor.jocor import JoCoR
from noisypy.methods.learning_strategies.co_teaching.utils import alpha_schedule
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config


class jocor_cifar10_clean_config(MethodConfig):

    data_config = cifar10_base_config

    classifier = resnet34
    classifier_args = dict(
        num_classes=10,
        weights=None,
    )

    learning_strategy_cls = JoCoR
    learning_strategy_args = dict(
        forget_rate=0.2,
        exponent=1,
        num_gradual=10,
        co_lambda=0.9,
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
        logger=CSVLogger("../logs", name="jocor_cifar10_clean"),
    )

    seed = 1337