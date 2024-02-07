from src.data.datasets.cifar100 import CIFAR100
from PIL import Image


def test_getitem(shared_tmp_path):
    dataset = CIFAR100(shared_tmp_path, download=True)
    sample = dataset[0]

    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)

def test_num_classes(shared_tmp_path):
    dataset = CIFAR100(shared_tmp_path, download=True)

    assert dataset.num_classes == 100
