import torch
from torch.utils.data import DataLoader
from PIL import Image

from .cifar import NoisyCIFAR10Dataset, CIFAR10DataModule, CIFAR10


class DivideMixNoisyCIFAR10Dataset(NoisyCIFAR10Dataset):
    def __init__(self, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.labeled_indices = torch.arange(self.data.shape[0])
        self.unlabeled_indices = torch.arange(self.data.shape[0])
        self.prediction = torch.ones(self.data.shape[0])
        self.probabilities = torch.ones(self.data.shape[0])

    def __len__(self):
        if self.mode == 'all':
            return len(self.data)
        elif self.mode == 'labeled':
            return len(self.labeled_indices)
        elif self.mode == 'unlabeled':
            return len(self.unlabeled_indices)
        else:
            raise ValueError("Invalid mode")


    def __getitem__(self, index):
        if self.mode == 'all':
            img = self.data[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            noisy_target = self.noisy_targets[index]
            return img, noisy_target, index
        elif self.mode == 'labeled':
            indices = self.labeled_indices[index]
            img = self.data[indices]
            img = Image.fromarray(img)
            prob = self.probabilities[indices]
            noisy_target = self.noisy_targets[indices]
            img1, img2 = self.transform(img), self.transform(img)
            return img1, img2, noisy_target, prob
        elif self.mode == 'unlabeled':
            indices = self.unlabeled_indices[index]
            img = self.data[indices]
            img = Image.fromarray(img)
            img1, img2 = self.transform(img), self.transform(img)
            return img1, img2
        else:
            raise ValueError("Invalid mode")


class DivideMixCIFAR10DataModule(CIFAR10DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.eval_train_dataset = DivideMixNoisyCIFAR10Dataset('all', self.noise_rate, self.noise_type, root=self.data_dir, train=True, transform=self.test_transform)
            self.train_dataset1 = DivideMixNoisyCIFAR10Dataset('all', self.noise_rate, self.noise_type, root=self.data_dir, train=True, transform=self.train_transform)
            self.train_dataset2 = DivideMixNoisyCIFAR10Dataset('all', self.noise_rate, self.noise_type, root=self.data_dir, train=True, transform=self.train_transform)
            self.unlabeled_dataset1 = DivideMixNoisyCIFAR10Dataset('unlabeled', self.noise_rate, self.noise_type, root=self.data_dir, train=True, transform=self.train_transform)
            self.unlabeled_dataset2 = DivideMixNoisyCIFAR10Dataset('unlabeled', self.noise_rate, self.noise_type, root=self.data_dir, train=True, transform=self.train_transform)
            # fix the noisy targets for all datasets
            self.train_dataset1.noisy_targets = self.eval_train_dataset.noisy_targets
            self.train_dataset2.noisy_targets = self.eval_train_dataset.noisy_targets
            self.unlabeled_dataset1.noisy_targets = self.eval_train_dataset.noisy_targets
            self.unlabeled_dataset2.noisy_targets = self.eval_train_dataset.noisy_targets
            assert self.train_dataset1.noisy_targets == self.train_dataset2.noisy_targets
            assert self.unlabeled_dataset1.noisy_targets == self.unlabeled_dataset2.noisy_targets
            assert self.train_dataset1.noisy_targets == self.unlabeled_dataset1.noisy_targets
            # Test dataset has no noise so use the original CIFAR10 class!
            self.test_dataset = CIFAR10(root=self.data_dir, train=False, transform=self.test_transform)


    def end_warmup(self):
        self.train_dataset1.mode = 'labeled'
        self.train_dataset2.mode = 'labeled'
        
        
    def set_probabilities(self, prob1, prob2):
        self.train_dataset1.probabilities = prob1
        self.train_dataset2.probabilities = prob2
        self.unlabeled_dataset1.probabilities = prob1
        self.unlabeled_dataset2.probabilities = prob2


    def set_predictions(self, pred1, pred2):
        self.train_dataset1.prediction = pred1
        self.train_dataset2.prediction = pred2
        labeled_indices1 = torch.arange(self.num_training_samples)[pred1]
        unlabeled_indices1 = torch.arange(self.num_training_samples)[~pred1]
        labeled_indices2 = torch.arange(self.num_training_samples)[pred2]
        unlabeled_indices2 = torch.arange(self.num_training_samples)[~pred2]
        self.train_dataset1.labeled_indices = labeled_indices1
        self.train_dataset1.unlabeled_indices = unlabeled_indices1
        self.train_dataset2.labeled_indices = labeled_indices2
        self.train_dataset2.unlabeled_indices = unlabeled_indices2
        # set unlabeled dataset indices
        self.unlabeled_dataset1.unlabeled_indices = unlabeled_indices1
        self.unlabeled_dataset2.unlabeled_indices = unlabeled_indices2


    def train_dataloader(self):
        dataloaders = {
            'warmup1': DataLoader(self.train_dataset1, batch_size=self.batch_size * 2, shuffle=True, num_workers=self.num_workers),
            'warmup2': DataLoader(self.train_dataset2, batch_size=self.batch_size * 2, shuffle=True, num_workers=self.num_workers),
            'labeled1': DataLoader(self.train_dataset1, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers),
            'labeled2': DataLoader(self.train_dataset2, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers),
            'unlabeled1': DataLoader(self.unlabeled_dataset1, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers),
            'unlabeled2': DataLoader(self.unlabeled_dataset2, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers),
            'eval_train': DataLoader(self.eval_train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
        }
        return dataloaders
