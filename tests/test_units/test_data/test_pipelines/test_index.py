from PIL import Image
from data.pipelines.index import AddIndex

from .common import CIFAR10, CIFAR10WithExtras, args

def test_simple():
    pipe = AddIndex()
    dataset_cls = pipe(CIFAR10)
    dataset = dataset_cls(**args)
    sample = dataset[0]

    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)
    assert isinstance(sample[2], int)


    assert dataset[0][2] == 0
    assert dataset[1337][2] == 1337

def test_extras():
    pipe = AddIndex()
    dataset_cls = pipe(CIFAR10WithExtras)
    dataset = dataset_cls(**args)
    sample = dataset[0]

    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)
    assert isinstance(sample[2], int)
    assert sample[3] == "something"
    assert sample[4] == "something_else"

    assert dataset[0][2] == 0
    assert dataset[1337][2] == 1337