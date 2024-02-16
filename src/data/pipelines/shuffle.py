import torch

from .base import AugmentationPipeline
from ..datasets.base import DatasetFW

# TODO: test if num_workers breaks this somehow
# TODO: implement which thing in __getitem__ to shuffle

class ShuffleSamples(AugmentationPipeline):
    # shuffles images within the dataset

    def __init__(self) -> None:
        super().__init__()
        self.shuffled_idxs = None
   
    def transform(self, dataset_cls: type[DatasetFW]) -> type[DatasetFW]:
        self_ = self

        class ShuffledDataset(dataset_cls):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if self_.shuffled_idxs is None:
                    self_.shuffled_idxs = torch.randperm(len(self))
                self.shuffled_idxs = self_.shuffled_idxs

            def __getitem__(self, index):
                (_, y, *other) = super().__getitem__(index)
                (x_shuffled, *_) = super().__getitem__(self.shuffled_idxs[index])
                return x_shuffled, y, *other

        return ShuffledDataset