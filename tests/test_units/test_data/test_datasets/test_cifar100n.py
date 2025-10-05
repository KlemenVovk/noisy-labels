from noisypy.data.datasets.cifar100n import CIFAR100N
from PIL import Image


def test_getitem(shared_tmp_path):
    dataset = CIFAR100N(
        "clean_label",
        shared_tmp_path / "noisylabels",
        shared_tmp_path / "cifar",
        download=True,
    )
    sample = dataset[0]

    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)


def test_num_classes(shared_tmp_path):
    dataset = CIFAR100N(
        "clean_label",
        shared_tmp_path / "noisylabels",
        shared_tmp_path / "cifar",
        download=True,
    )

    assert dataset.num_classes == 100
