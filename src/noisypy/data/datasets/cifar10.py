from torchvision.datasets import CIFAR10 as CIFAR10PT
from torchvision import transforms

from .base import DatasetFW


class CIFAR10(CIFAR10PT, DatasetFW): # this is perhaps not very nice
    
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