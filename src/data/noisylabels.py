from typing import Literal, Dict
import requests
import os
from tqdm import tqdm
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import lightning as L


class NoisyLabels:
    """
    Handles downloading and loading of noisylabels targets
    """
    # TODO: check file integrity before skipping download (partial downloads break downloading currently)

    _name_map = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
    }

    @classmethod
    def get_filename(cls, dataset: Literal["cifar10", "cifar100"]):
        return f"{cls._name_map[dataset]}_human.pt"

    @classmethod
    def get_url(cls, dataset: Literal["cifar10", "cifar100"]):
        return f"https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/{cls.get_filename(dataset)}"

    def __init__(self, dataset: Literal["cifar10", "cifar100"], dir: str, download: bool) -> None:
        self.dataset = dataset
        self.save_path = Path(dir) / self.get_filename(dataset)
        if download: self.download()

    def load(self) -> Dict[str, Tensor]:
        if self.save_path.exists():
            with open(self.save_path, "rb") as f:
                noise_dict = torch.load(f)
            return noise_dict
        else:
            raise ValueError

    def download(self) -> None:
        """
        Downloads noisylabel labels from official github repo
        """
        if self.save_path.exists():
            print("File already downloaded.")
            return

        url = self.get_url(self.dataset)
        response = requests.get(url, stream=True)

        os.makedirs(self.save_path.parent, exist_ok=True)
        total = 2401007 if self.dataset == "cifar10" else 1000959
        with open(self.save_path, "wb") as fo:
            for data in tqdm(response.iter_content(), total=total):
                fo.write(data)


class NoisyLabelsCIFAR10Dataset(CIFAR10):

    def __init__(
            self, 
            noise_type: Literal[
                "clean_label",
                "aggre_label",
                "worse_label",
                "random_label1",
                "random_label2",
                "random_label",
                ],
            noise_dir: str,
            cifar_dir: str,
            download: bool,
            *args, **kwargs,
            ):
        super().__init__(cifar_dir, *args, download=download, train=True, **kwargs)
        noise_dict = NoisyLabels("cifar10", noise_dir, download).load()
        self.noisy_targets = noise_dict[noise_type]
        
    def __getitem__(self, index):
        img, true_target = super().__getitem__(index)
        return img, self.noisy_targets[index], true_target
    

class NoisyLabelsCIFAR100Dataset(CIFAR100):

    def __init__(
            self, 
            noise_type: Literal[
                "clean_label",
                "noisy_label",
                "noisy_coarse_label",
                "clean_coarse_label",
                ],
            noise_dir: str,
            cifar_dir: str,
            download: bool,
            *args, **kwargs,
            ):
        super().__init__(cifar_dir, *args, download=download, train=True, **kwargs)
        noise_dict = NoisyLabels("cifar100", noise_dir, download).load()
        self.noisy_targets = noise_dict[noise_type]
        
    def __getitem__(self, index):
        img, true_target = super().__getitem__(index)
        return img, self.noisy_targets[index], true_target
    

class NoisylabelsDataModule(L.LightningDataModule):

    @property
    def num_classes(self):
        return 10 if self.dataset_name == "cifar10" else 100
    
    @property
    def num_training_samples(self):
        return 50000
    
    def __init__(
            self, train_transform, 
            test_transform, 
            dataset_name: Literal["cifar10", "cifar100"] = "cifar10",
            noise_type="clean_label", # refer to cifar10 or 100 noise type args
            cifar_dir="../data/cifar",
            noisylabels_dir="../data/noisylabels",
            batch_size=64,
            num_workers=4,
            ):
        super().__init__()
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.dataset_name = dataset_name
        self.noise_type = noise_type
        self.cifar_dir = cifar_dir
        self.noisylabels_dir = noisylabels_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # Download CIFAR and noisylabels data if needed
        NoisyLabels(self.dataset_name, self.noisylabels_dir, True)
        if self.dataset_name == "cifar10":
            CIFAR10(root=self.cifar_dir, train=False, download=True)
        else:
            CIFAR100(root=self.cifar_dir, train=False, download=True)

    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.dataset_name == "cifar10":
                self.train_dataset = NoisyLabelsCIFAR10Dataset(self.noise_type, self.noisylabels_dir, self.cifar_dir, False, transform=self.test_transform)
                self.test_dataset = CIFAR10(self.cifar_dir, train=False, transform=self.test_transform)
            else:
                self.train_dataset = NoisyLabelsCIFAR100Dataset(self.noise_type, self.noisylabels_dir, self.cifar_dir, False, transform=self.test_transform)
                self.test_dataset = CIFAR100(self.cifar_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)