from pytorch_lightning.loggers import CSVLogger

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import ToTensor

from noisypy.methods.learning_strategies.co_teaching.co_teaching import CoTeachingPlus
from noisypy.methods.learning_strategies.co_teaching.utils import alpha_schedule
from .utils import CNN_small
from noisypy.configs.base.method import MethodConfig
from noisypy.data.pipelines.base import Compose
from noisypy.data.pipelines.index import AddIndex
from noisypy.data.pipelines.noise.noises import SymmetricNoise, InstanceNoise
from noisypy.data.pipelines.noise.pipeline import AddNoise

from ..common import cifar10_base_config

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
        num_workers=2
    )

class co_teaching_plus_reprod(MethodConfig):

    data_config = cifar10_noise_index

    classifier = CNN_small
    classifier_args = dict(
        num_classes=10
    )

    learning_strategy_cls = CoTeachingPlus
    learning_strategy_args = dict(
        forget_rate=0.2,
        exponent=1,
        num_gradual=10,
        num_epochs=200,
        init_epoch=20,
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
        logger=CSVLogger("../logs", name="CoTeachingPlus")
    )

    seed = 1