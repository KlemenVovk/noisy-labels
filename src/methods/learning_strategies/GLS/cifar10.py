from typing import Callable, Any
from torchvision.datasets import CIFAR10 as CIFAR10PT
from torchvision import transforms

from PIL import Image
from data.datasets.base import DatasetFW
import torch

train_data = "/home/klemen/projects/negative-label-smoothing/traindata_0.2.pt"
test_data = "/home/klemen/projects/negative-label-smoothing/testdata.pt"
train_data = torch.load(train_data)
test_data = torch.load(test_data)

def lambda_gls_noise(feature, target, index):
    return train_data["noisy_labels"][index]


class GLSCIFAR10(CIFAR10PT, DatasetFW): # this is perhaps not very nice
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        if self.train:
            img, target = train_data["images"][index], train_data["clean_labels"][index].squeeze()
        else:
            img, target = test_data["images"][index], test_data["labels"][index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @property
    def num_classes(self) -> int:
        return 10

