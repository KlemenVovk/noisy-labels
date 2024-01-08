from src.data.datasets.noisylabels import NoisyLabelsLoader
import os
import shutil

from .common import noisylabels_save_path

tmp_dir = "noisylabels_test_tmp"

def test_cifar10_load_all():
    nl = NoisyLabelsLoader("cifar10", noisylabels_save_path, False)
    labels = nl.load_all()

    assert isinstance(labels, dict)
    for key in nl.cifar10_label_names:
        assert key in labels

def test_cifar10_load_label():
    nl = NoisyLabelsLoader("cifar10", noisylabels_save_path, False)
    
    for key in nl.cifar10_label_names:
        nl.load_label(key)

def test_cifar100_load_all():
    nl = NoisyLabelsLoader("cifar100", noisylabels_save_path, False)
    labels = nl.load_all()

    assert isinstance(labels, dict)
    for key in nl.cifar100_label_names:
        assert key in labels

def test_cifar100_load_label():
    nl = NoisyLabelsLoader("cifar100", noisylabels_save_path, False)
    
    for key in nl.cifar100_label_names:
        nl.load_label(key)

def test_download():
    nl = NoisyLabelsLoader("cifar10", tmp_dir, True)
    nl = NoisyLabelsLoader("cifar10", tmp_dir, False)
    nl.load_all()
    shutil.rmtree(tmp_dir)

def test_integrity_check():
    nl = NoisyLabelsLoader("cifar10", tmp_dir, True)
    with open(nl.save_path, "w") as f:
        f.write("hhh")
    nl = NoisyLabelsLoader("cifar10", tmp_dir, True)
    nl.load_all()
    shutil.rmtree(tmp_dir)