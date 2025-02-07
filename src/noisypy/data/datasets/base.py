from abc import abstractmethod, ABC
from typing import Any

from torch.utils.data.dataset import Dataset as DatasetPT
from torch import Tensor


class DatasetFW(DatasetPT, ABC):
    """Abstract base class for datasets in this (F)rame(W)ork.
    Extends the pytorch dataset.
    Besides implementing the __getitem__ and __len__ required by the
    pytorch dataset, the framework requires also implementation of:
    
    - num_classes property: Number of classification label classes.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Tensor, int, Any]:
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
      
    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError