from typing import Callable, Any
from torchvision.datasets import CIFAR100 as CIFAR100PT
from torchvision import transforms

from .base import DatasetFW


class CIFAR100(CIFAR100PT, DatasetFW): # yeah its basically a copy of CIFAR10 :)
    
    def setup(self) -> None:
        self.download()

    @property
    def num_classes(self) -> int:
        return 100

# from noisylabels
train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])