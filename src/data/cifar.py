import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from data.synthetic_noise import generate_instance_dependent_noise
import lightning as L

# TODO: transforms
# TODO: val dataset

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, train_transform, data_dir='../data/cifar10', batch_size=32, num_workers=8, noise_rate=0.2, add_synthetic_noise=False):
        super().__init__()
        self.train_transform = train_transform
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_rate = noise_rate
        self.num_classes = 10
        self.add_synthetic_noise = add_synthetic_noise
    
    def prepare_data(self):
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CIFAR10(root=self.data_dir, train=True, transform=self.train_transform)
            if self.add_synthetic_noise: # add noise to train dataset
                print('Adding noise to train dataset')
                y = torch.tensor(self.train_dataset.targets)
                y_noisy = generate_instance_dependent_noise(self.train_dataset.data, y, self.noise_rate, num_classes=self.num_classes)
                print('Noise rate:', (y != y_noisy).sum().item() / len(y))
                self.train_dataset.targets = y_noisy
            self.test_dataset = CIFAR10(root=self.data_dir, train=False, transform=transforms.ToTensor())

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
