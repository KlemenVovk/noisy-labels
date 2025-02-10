from typing import Type
from ..datasets.base import DatasetFW
from .base import AugmentationPipeline


class DoubleAugmentation(AugmentationPipeline):
    def __init__(self, transform1, transform2) -> None:
        super().__init__()
        self.transform1 = transform1
        self.transform2 = transform2

    def transform(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:
        transform1 = self.transform1
        transform2 = self.transform2
        class DoubleAugmentationDataset(dataset_cls):
            # NOTE: transforms should be none for the original dataset
            # currently done in data config, but could be ensured here?
            def __init__ (self, *dataset_args, **dataset_kwargs):
                self.initialized = False
                super().__init__(*dataset_args, **dataset_kwargs)
                self.transform1 = transform1
                self.transform2 = transform2
                self.initialized = True

            def __getitem__(self, index):
                if not self.initialized:
                    return super().__getitem__(index)
                (x1, y, *other) = super().__getitem__(index)
                (x2, _, *_) = super().__getitem__(index)
                return self.transform1(x1), self.transform2(x2), y, *other

        return DoubleAugmentationDataset