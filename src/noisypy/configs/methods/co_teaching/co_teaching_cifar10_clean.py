from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch.loggers import CSVLogger

from noisypy.methods.classifiers.resnet import resnet34
from noisypy.methods.learning_strategies.co_teaching.co_teaching import CoTeaching
from noisypy.methods.learning_strategies.co_teaching.utils import alpha_schedule
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config
from noisypy.data.pipelines.index import AddIndex


class cifar10_clean_index_config(cifar10_base_config):

    dataset_train_augmentation = AddIndex()


class co_teaching_cifar10_clean_config(MethodConfig):

    data_config = cifar10_clean_index_config

    classifier = resnet34
    classifier_args = dict(
        num_classes=10,
        weights=None,
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
        logger=CSVLogger("../logs", name="co_teaching_cifar10_clean")
    )

    seed = 1337