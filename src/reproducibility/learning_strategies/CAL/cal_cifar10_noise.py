from pytorch_lightning.loggers import CSVLogger
import torch

from noisypy.methods.learning_strategies.CAL.utils import SegAlpha
from noisypy.data.pipelines.base import Compose
from noisypy.data.pipelines.index import AddIndex
from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.noise.noises import InstanceNoise
from .cal_cifar10_clean import cal_cifar10_clean
from ..common import cifar10_base_config

class cifar10_index_noise_config(cifar10_base_config):
    dataset_train_augmentation = Compose([
        AddNoise(noise=InstanceNoise(torch.load("methods/learning_strategies/CAL/reproducibility/IDN_0.4_C10.pt"))), 
        AddIndex()]) 

class cal_cifar10_noise(cal_cifar10_clean):

    data_config = cifar10_index_noise_config

    learning_strategy_args = dict(alpha = 0.0, 
                                  warmup_epochs = 65,
                                  alpha_scheduler_cls = SegAlpha,
                                  alpha_scheduler_args = dict(
                                    alpha_list = [0.0, 1.0, 1.0],
                                    milestones = [10, 40, 80],
                                  ),
                                  alpha_scheduler_args_warmup = dict(
                                    alpha_list = [0.0, 2.0],
                                    milestones = [10, 40],
                                  ),
    ) 

    trainer_args = dict(
        max_epochs=165,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="cal_noise")
    )

    seed = 1337