from .base import AugmentationPipeline, D


class AddIndex(AugmentationPipeline):

    def transform(self, dataset_cls: D) -> D:

        class DatasetWithIndex(dataset_cls):

            def __getitem__(self, index):
                (x, y, *other) = super().__getitem__(index)
                return x, y, index, *other

        return DatasetWithIndex