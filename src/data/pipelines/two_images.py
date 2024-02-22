from typing import Type
from ..datasets.base import DatasetFW
from .base import AugmentationPipeline


class TwoImages(AugmentationPipeline):

    def transform(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:

        class DatasetWithTwoImages(dataset_cls):

            def __getitem__(self, index):
                (x1, y, *other) = super().__getitem__(index)
                (x2, _, *_) = super().__getitem__(index)
                return x1, x2, y, *other

        return DatasetWithTwoImages