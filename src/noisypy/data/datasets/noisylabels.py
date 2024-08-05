from typing import Literal, Dict
import requests
import os
from pathlib import Path
from tqdm import tqdm

import torch
from torch import Tensor
from torchvision.datasets.utils import check_integrity


class NoisyLabelsLoader:
    """
    Handles downloading and loading of noisylabels targets
    """

    _name_map = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
    }

    _md5_map = {
        "cifar10": "96644db5574813cae939af5f7f2a0aec",
        "cifar100": "a67b45c1d7992051865b5c755bd68bb1",
    }

    cifar10_label_names = [
        "clean_label", "aggre_label", "worse_label",
        "random_label1", "random_label2", "random_label3"
    ]

    cifar100_label_names = [
        "clean_label", "noisy_label",
        "noisy_coarse_label", "clean_coarse_label"
    ]

    @classmethod
    def get_filename(cls, dataset: Literal["cifar10", "cifar100"]) -> str:
        return f"{cls._name_map[dataset]}_human.pt"

    @classmethod
    def get_url(cls, dataset: Literal["cifar10", "cifar100"]) -> str:
        return f"https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/{cls.get_filename(dataset)}"

    def __init__(self, dataset: Literal["cifar10", "cifar100"], dir: str, download: bool) -> None:
        self.dataset = dataset
        self.save_path = Path(dir) / self.get_filename(dataset)
        if download: self.download()

    def load_all(self) -> Dict[str, Tensor]:
        if self.save_path.exists():
            with open(self.save_path, "rb") as f:
                noise_dict = torch.load(f)
            return noise_dict
        else:
            raise ValueError(f"Cannot find label file at {self.save_path}.")
    
    def load_label(
            self, 
            label_name = Literal["clean_label", "aggre_label", "worse_label",
                                 "random_label1", "random_label2", "random_label3"] # if cifar-10
                        | Literal["clean_label", "noisy_label",
                                  "noisy_coarse_label", "clean_coarse_label"] # if cifar-100
            ) -> Tensor:
        noise_dict = self.load_all()
        return noise_dict[label_name]

    def download(self) -> None:
        """
        Downloads noisylabel labels from official github repo
        """
        if self.save_path.exists() and self._check_integrity():
            print("Label file already downloaded and verified.")
            return

        url = self.get_url(self.dataset)
        response = requests.get(url, stream=True)

        os.makedirs(self.save_path.parent, exist_ok=True)
        total = 2401007 if self.dataset == "cifar10" else 1000959
        with open(self.save_path, "wb") as fo:
            for data in tqdm(response.iter_content(), total=total):
                fo.write(data)

    def _check_integrity(self):
        return check_integrity(self.save_path, self._md5_map[self.dataset])
