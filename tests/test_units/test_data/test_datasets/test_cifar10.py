from data.datasets.cifar10 import CIFAR10
from PIL import Image


def test_getitem(shared_tmp_path):
    dataset = CIFAR10(shared_tmp_path, download=True)
    sample = dataset[0]
    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)

def test_num_classes(shared_tmp_path):
    dataset = CIFAR10(shared_tmp_path, download=True)
    assert dataset.num_classes == 10
