from src.data.datasets.cifar100 import CIFAR100
from PIL import Image

from .common import cifar_save_path

def test_getitem():
    dataset = CIFAR100(cifar_save_path, download=True)
    sample = dataset[0]

    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)

def test_num_classes():
    dataset = CIFAR100(cifar_save_path, download=True)

    assert dataset.num_classes == 100
