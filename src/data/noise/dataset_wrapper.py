from .noises import Noise
from ..datasets.base import Dataset


def noisify_dataset(cls: Dataset, noise: Noise) -> Dataset:

    class NoisyDataset(cls):

        def __init__ (self, *dataset_args, **dataset_kwargs):
            super().__init__(*dataset_args, **dataset_kwargs)
            self.noise = noise

        def __getitem__(self, index):
            # add noise to targets
            (x, y, *other) = super().__getitem__(index)
            return x, self.noise(x, y, index), *other
    
    return NoisyDataset