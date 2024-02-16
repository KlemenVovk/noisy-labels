import torch
from torch import Tensor

from .noise.noises import Noise
from .base import AugmentationPipeline


class DivideMixify(AugmentationPipeline):
    """
        DivideMix implementation based on https://openreview.net/pdf?id=HJgExaVtwr
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, dataset_cls):

        class DivideMixCIFAR10Dataset(dataset_cls):
            def __init__(self, mode, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.mode = mode
                self.labeled_indices = torch.arange(self.data.shape[0])
                self.unlabeled_indices = torch.arange(self.data.shape[0])
                self.prediction = torch.ones(self.data.shape[0])
                self.probabilities = torch.ones(self.data.shape[0])

            def __len__(self):
                if self.mode == 'all':
                    return super().__len__()
                elif self.mode == 'labeled':
                    return len(self.labeled_indices)
                elif self.mode == 'unlabeled':
                    return len(self.unlabeled_indices)
                else:
                    raise ValueError("Invalid mode")


            def __getitem__(self, index):
                if self.mode == 'all':
                    sample = super().__getitem__(index)
                    return *sample, index
                elif self.mode == 'labeled':
                    indices = self.labeled_indices[index]
                    prob = self.probabilities[indices]
                    sample1 = super().__getitem__(indices)
                    sample2 = super().__getitem__(indices)
                    return sample1[0], sample2[0], sample1[-1], prob
                elif self.mode == 'unlabeled':
                    indices = self.unlabeled_indices[index]
                    sample1 = super().__getitem__(indices)
                    sample2 = super().__getitem__(indices)
                    return sample1[0], sample2[0]
                else:
                    raise ValueError("Invalid mode")
                
            def end_warmup(self):
                self.train_dataset1.mode = 'labeled'
                self.train_dataset2.mode = 'labeled'
                
        return DivideMixCIFAR10Dataset


class DivideMixSymmetricNoise(Noise):
    """
        Symmetric noise implementation based on 
        https://github.com/LiJunnan1992/DivideMix/blob/d9d3058fa69a952463b896f84730378cdee6ec39/dataloader_cifar.py#L58-L78
        This is different from our implementation where the expected noise rate is noise_rate,
        whereas here the expected noise rate is noise_rate - 1/num_classes * noise_rate.
    """
    def __init__(self, noise_rate, num_samples, num_classes) -> None:
        super().__init__()
        idx = torch.randperm(num_samples)
        num_noise = int(noise_rate*num_samples)
        self.noise_idx = idx[:num_noise]
        self.num_classes = num_classes

    def _noisify_target(self, feature: Tensor, target: int | Tensor, index: int | Tensor) -> int:
        if index in self.noise_idx:
            return torch.randint(0, self.num_classes, (1,1)).item()
        return target