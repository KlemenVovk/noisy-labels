from .noises import Noise
from ..base import AugmentationPipeline


class AddNoise(AugmentationPipeline):
    """Augments the dataset by transforming labels with Noise.

    Args:
        noise (Noise): Noise to transform the labels with.
        keep_original (bool): Whether to return original label as well or not.
    """

    def __init__(self, noise: Noise, keep_original: bool = False) -> None:
        super().__init__()
        self.noise = noise
        self.keep_original = keep_original

    def transform(self, dataset_cls):
        noise_ = self.noise
        keep_ = self.keep_original

        class NoisyDataset(dataset_cls):
            def __init__(self, *dataset_args, **dataset_kwargs):
                super().__init__(*dataset_args, **dataset_kwargs)
                self.noise = noise_

                # precache noise so its consitent between workers
                for i in range(len(self)):
                    self[i]

            def __getitem__(self, index):
                # add noise to targets
                (x, y, *other) = super().__getitem__(index)
                if keep_:
                    return x, self.noise(x, y, index), y, *other
                return x, self.noise(x, y, index), *other

        return NoisyDataset
