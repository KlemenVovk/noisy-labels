from pytorch_lightning.loggers import CSVLogger

from data.pipelines.base import Compose
from data.pipelines.divide_mix import DivideMixify
from data.pipelines.noise.pipeline import AddNoise
from data.pipelines.noise.noises import SymmetricNoise
from configs.methods.divide_mix.divide_mix_cifar10_clean import divide_mix_cifar10_base_config, divide_mix_cifar10_clean


noise_rate = 0.5


class divide_mix_cifar10_noise_config(divide_mix_cifar10_base_config):
    dataset_train_augmentation = Compose([
        AddNoise(noise=SymmetricNoise(num_classes=10, noise_rate=noise_rate)), 
        DivideMixify()])

    
class divide_mix_cifar10_noise(divide_mix_cifar10_clean):

    learning_strategy_args = dict(warmup_epochs=10, 
                                  noise_type = "symmetric", 
                                  noise_rate = noise_rate,
                                  p_thresh = 0.5, 
                                  temperature = 0.5, 
                                  alpha = 4)

    trainer_args = dict(
        max_epochs=300,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="divide_mix_symmetric")
    )

    seed = 1337