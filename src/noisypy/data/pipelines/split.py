from typing import Any, Type

from torch import Tensor
from ..datasets.base import DatasetFW
from .base import AugmentationPipeline


# NOTE: cannot be used in compose unless at the end of the pipeline


class Split(AugmentationPipeline):
    def __init__(self, train_size: float) -> None:
        super().__init__()
        self.ratio = train_size

    def transform(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:
        r = self.ratio

        class TrainDataset(dataset_cls):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                n = super().__len__()
                cut = int(r * n)
                self.valid_idxs = list(range(cut))

            def __len__(self) -> int:
                return len(self.valid_idxs)

        class ValDataset(dataset_cls):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                n = super().__len__()
                cut = int(r * n)
                self.valid_idxs = list(range(cut, n))

            def __getitem__(self, index: int) -> tuple[Tensor, int, Any]:
                return super().__getitem__(self.valid_idxs[index])

            def __len__(self) -> int:
                return len(self.valid_idxs)

        return TrainDataset, ValDataset
