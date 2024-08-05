from aim.pytorch_lightning import AimLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import ToTensor

from noisypy.methods.learning_strategies.jocor.jocor import JoCoR
from .utils import CNN
from noisypy.methods.learning_strategies.co_teaching.utils import alpha_schedule
from noisypy.configs.base.method import MethodConfig
from noisypy.data.pipelines.noise.noises import SymmetricNoise
from noisypy.data.pipelines.noise.pipeline import AddNoise

from ..common import cifar10_base_config


# reproduces the original implementation, with their noise and all

class cifar10_noise(cifar10_base_config):

    dataset_train_augmentation = AddNoise(SymmetricNoise(10, 0.2))

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
        num_workers=2
    )

class jocor_reprod(MethodConfig):

    data_config = cifar10_noise

    classifier = CNN
    classifier_args = dict(
        input_channel=3,
        n_outputs=10
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
        logger=AimLogger(experiment="JoCoR")
    )

    seed = 1