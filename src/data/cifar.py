import torch
from torchvision.datasets import CIFAR10
from utils.noise_generators import generate_instance_dependent_noise, noisify_instance
import lightning as L

# TODO: add real validation set (currently we are using the test set as validation, just because validation is automatically run after each epoch so we can see progress in real time)
# TODO: switch to own implementation of instance-dependent noise generation
# TODO: would be a good idea to log a few images (before and after transform) so that we can view them in aim and make sure they are correct

class NoisyCIFAR10Dataset(CIFAR10):
    def __init__(self, noise_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_rate = noise_rate
        # Instance dependent noise! # TODO: add symmetric and asymetric noise
        self.noisy_targets, _ = noisify_instance(self.data, self.targets, self.noise_rate)

    def __getitem__(self, index):
        img, true_target = super().__getitem__(index)
        return img, self.noisy_targets[index], true_target



class CIFAR10DataModule(L.LightningDataModule):

    @property
    def num_classes(self):
        return 10
    
    @property
    def num_training_samples(self):
        return 50000

    def __init__(self, train_transform, test_transform, data_dir='../data/cifar10', batch_size=64, num_workers=4, noise_rate=0.2, noise_type=None):
        # TODO: different noise types
        super().__init__()
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_rate = noise_rate
        self.noise_type = noise_type

    def prepare_data(self):
        # Just to download the dataset
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = NoisyCIFAR10Dataset(self.noise_rate, root=self.data_dir, train=True, transform=self.train_transform)
            # Test dataset has no noise so use the original CIFAR10 class!
            self.test_dataset = CIFAR10(root=self.data_dir, train=False, transform=self.test_transform)
    
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)