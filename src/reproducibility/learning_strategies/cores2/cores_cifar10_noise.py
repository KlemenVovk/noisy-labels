from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.noise.noises import SymmetricNoise

from .cores_cifar10_clean import cores_cifar10_clean

from ..common import cifar10_base_config


class cifar10_noise(cifar10_base_config):
    dataset_train_augmentation = AddNoise(
        noise=SymmetricNoise(num_classes=10, noise_rate=0.3)
    )


class cores_cifar10_noise(cores_cifar10_clean):
    data_config = cifar10_noise
