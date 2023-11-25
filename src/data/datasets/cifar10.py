from typing import Callable, Any
from torchvision.datasets import CIFAR10 as CIFAR10PT

from .base import Dataset
from .registry import DATASETS


@DATASETS.register_module(name="cifar10")
class CIFAR10(CIFAR10PT, Dataset): # this is perhaps not very nice
    
    def __init__(self, 
                 root: str, 
                 train: bool = True, 
                 transform: Callable[..., Any] | None = None, 
                 target_transform: Callable[..., Any] | None = None) -> None:
        super().__init__(root, train, transform, target_transform, download=False)

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()
    
    def setup(self) -> None:
        self.download()

    @property
    def num_classes(self) -> int:
        return 10