import torch
from PIL import Image

from noisypy.data.pipelines.base import Compose, Identity

from noisypy.data.pipelines.noise.pipeline import AddNoise
from noisypy.data.pipelines.noise.noises import InstanceNoise
from noisypy.data.pipelines.index import AddIndex

from .common import CIFAR10, CIFAR10WithExtras, args

def test_compose():
    pipe = Compose([
        AddIndex(),
        AddNoise(InstanceNoise(torch.arange(1, 50001)))
    ])
    dataset_cls = pipe(CIFAR10WithExtras)
    dataset = dataset_cls(**args)
    sample = dataset[0]

    # check types
    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)
    assert isinstance(sample[2], int)

    # check extras
    assert sample[3] == "something"
    assert sample[4] == "something_else"

    # check index
    assert dataset[0][2] == 0
    assert dataset[2][2] == 2

    # check noise
    for i in range(3):
        assert dataset[i][1] == i+1

def test_identity():
    pipe = Identity()
    dataset_cls = pipe(CIFAR10)
    dataset_orig = CIFAR10(**args)
    dataset_pipe = dataset_cls(**args)
    sample = dataset_pipe[0]

    # check types
    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)

    # check identity
    for i in range(10):
        assert dataset_pipe[i] == dataset_orig[i]