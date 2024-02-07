from typing import Callable, Any
from pathlib import Path
import requests

from torchvision.datasets import EuroSAT as EuroSATPT
from torchvision import transforms

from .base import DatasetFW


# TODO: download fails because some ssl certificate or sth

class EuroSAT(EuroSATPT, DatasetFW):

    _split_url = "https://storage.googleapis.com/remote_sensing_representations/eurosat-test.txt"
    _split_name = "eurosat-test.txt"

    def __init__(self, 
                 root: str,
                 train: bool = True,
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None,
                 download: bool = False) -> None:
        super().__init__(root, transform, target_transform, download)

        root_path = Path(root)

        # download test split
        if download:
            response = requests.get(self._split_url)
            with open(root_path / self._split_name, "wb") as f:
                f.write(response.content)

        # load test samples' filenames
        with open(root_path / self._split_name, "r") as f:
            test_fnames = f.read().splitlines()

        # override samples with only the ones from train/test set
        def in_split(fpath: str) -> bool:
            fname = Path(fpath).name
            if train:
                return fname not in test_fnames
            return fname in test_fnames
        self.samples = [s for s in self.samples if in_split(s[0])]
    
    @property
    def num_classes(self) -> int:
        return 10

# TODO: transforms inline with other research
eurosat_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3444, 0.3803, 0.4078), (0.2027, 0.1369, 0.1156)),
])

eurosat_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3444, 0.3803, 0.4078), (0.2027, 0.1369, 0.1156)),
])