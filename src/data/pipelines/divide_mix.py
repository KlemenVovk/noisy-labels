import torch
from PIL import Image

from .base import AugmentationPipeline


class DivideMixify(AugmentationPipeline):
    """

    Args:
        noise (Noise): Noise to transform the labels with.
        keep_original (bool): Whether to return original label as well or not.
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
