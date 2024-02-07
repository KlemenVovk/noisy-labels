from src.data.datasets.eurosat import EuroSAT
from PIL import Image
from pathlib import Path


def test_getitem(shared_tmp_path):
    dataset = EuroSAT(shared_tmp_path, True, None, None, True)
    sample = dataset[0]

    assert isinstance(sample[0], Image.Image)
    assert isinstance(sample[1], int)

def test_num_classes(shared_tmp_path):
    dataset = EuroSAT(shared_tmp_path, True, None, None, True)
    
    assert dataset.num_classes == 10

def test_train_split(shared_tmp_path):
    dataset = EuroSAT(shared_tmp_path, True, None, None, True)
    test_path = shared_tmp_path / dataset._split_name
    with open(test_path, "r") as f:
        test_fnames = f.read().splitlines()

    for s, _ in dataset.samples:
        assert Path(s).name not in test_fnames

def test_test_split(shared_tmp_path):
    dataset = EuroSAT(shared_tmp_path, False, True)
    test_path = shared_tmp_path / dataset._split_name
    with open(test_path, "r") as f:
        test_fnames = f.read().splitlines()

    for s, _ in dataset.samples:
        assert Path(s).name in test_fnames