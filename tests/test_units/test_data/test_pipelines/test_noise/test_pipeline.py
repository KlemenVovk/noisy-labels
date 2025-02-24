import torch
from PIL import Image

from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.noise.noises import InstanceNoise

from ..common import CIFAR10, CIFAR10WithExtras, args


def test_simple():
    pipe = AddNoise(InstanceNoise(torch.arange(1, 50001)))
    dataset_cls = pipe(CIFAR10)
    dataset = dataset_cls(**args)
    sample = dataset[0]

    # check types
    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)

    # check noise
    for i in range(3):
        assert dataset[i][1] == i + 1


def test_extras():
    pipe = AddNoise(InstanceNoise(torch.arange(1, 50001)))
    dataset_cls = pipe(CIFAR10WithExtras)
    dataset = dataset_cls(**args)
    sample = dataset[0]

    # check types
    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)

    # check extras
    assert sample[2] == "something"
    assert sample[3] == "something_else"

    # check noise
    for i in range(3):
        assert dataset[i][1] == i + 1


def test_persistance():
    pipe = AddNoise(InstanceNoise(torch.arange(1, 50001)))
    dataset_cls1 = pipe(CIFAR10)
    dataset_cls2 = pipe(CIFAR10)

    dataset1 = dataset_cls1(**args)
    dataset2 = dataset_cls2(**args)

    for i in range(3):
        assert dataset1[i][1] == dataset2[i][1] == i + 1
