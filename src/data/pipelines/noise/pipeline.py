from .noises import Noise
from ..base import AugmentationPipeline


class NoisePipeline(AugmentationPipeline):

    def __init__(self, noise: Noise) -> None:
        super().__init__()
        self.noise = noise

    def transform(self, dataset_cls):
        noise_ = self.noise

        class NoisyDataset(dataset_cls):

            def __init__ (self, *dataset_args, **dataset_kwargs):
                super().__init__(*dataset_args, **dataset_kwargs)
                self.noise = noise_

            def __getitem__(self, index):
                # add noise to targets
                (x, y, *other) = super().__getitem__(index)
                return x, self.noise(x, y, index), *other

        return NoisyDataset