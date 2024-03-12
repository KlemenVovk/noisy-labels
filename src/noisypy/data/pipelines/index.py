from typing import Type
from ..datasets.base import DatasetFW
from .base import AugmentationPipeline


class AddIndex(AugmentationPipeline):

    def transform(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:

        class DatasetWithIndex(dataset_cls):

            def __getitem__(self, index):
                (x, y, *other) = super().__getitem__(index)
                return x, y, index, *other

        return DatasetWithIndex