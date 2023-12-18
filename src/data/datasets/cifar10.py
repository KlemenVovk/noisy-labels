from typing import Callable, Any
from torchvision.datasets import CIFAR10 as CIFAR10PT
from torchvision import transforms

from .base import DatasetFW


class CIFAR10(CIFAR10PT, DatasetFW): # this is perhaps not very nice
    
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


cifar10_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])