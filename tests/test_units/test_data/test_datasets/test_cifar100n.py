from src.data.datasets.cifar100n import CIFAR100N
from PIL import Image

from .common import cifar_save_path, noisylabels_save_path

def test_getitem():
    dataset = CIFAR100N("clean_label", noisylabels_save_path, cifar_save_path)
    sample = dataset[0]

    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)

def test_num_classes():
    dataset = CIFAR100N("clean_label", noisylabels_save_path, cifar_save_path)
    
    assert dataset.num_classes == 100
