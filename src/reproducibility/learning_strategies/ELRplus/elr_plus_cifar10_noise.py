from pytorch_lightning.loggers import CSVLogger

from noisypy.data.pipelines.base import Compose
from noisypy.data.pipelines.index import AddIndex
from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.noise.noises import BiasedSymmetricNoise
from noisypy.configs.data.cifar10 import cifar10_base_config
from ..ELR.elr_cifar10_noise import cifar10_index_noise_config
from .elr_plus_cifar10_clean import elr_plus_cifar10_clean

class cifar10_index_noise_config(cifar10_base_config):

    dataset_train_augmentation = Compose([
        AddNoise(noise=BiasedSymmetricNoise(num_classes=10, noise_rate=0.8)), 
        AddIndex()])
        
class elr_plus_cifar10_noise(elr_plus_cifar10_clean):

    data_config = cifar10_index_noise_config

    learning_strategy_args = dict(beta = 0.7, # β ∈ {0.5, 0.7, 0.9, 0.99} beta in original paper
                                  lmbd=3,      # λ ∈ {1, 3, 5, 7, 10} lambda in original paper
                                  gamma=0.997, # γ ∈ [0, 1] ema_alpha in original paper
                                  alpha=1,     # α ∈ {0, 0.1, 1, 2, 5} mixup_alpha in original paper
                                  ema_update=True, # True or False
                                  ema_step=40000,  # EMA step (in iterations)
                                  coef_step=0 
    )

    trainer_args = dict(
        max_epochs=200 * 2,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="elr_plus_symmetric")
    )

    seed = 1337