from abc import abstractmethod, ABC
from typing import Tuple, Any

from torch.utils.data.dataset import Dataset as DatasetPT
from torch import Tensor

# TODO setup needs to be changed somehow as it is currently broken

class DatasetFW(DatasetPT, ABC):
    """Abstract base class for datasets in this (F)rame(W)ork.
    Extends the pytorch dataset.
    Besides implementing the __getitem__ and __len__ required by the
    pytorch dataset, the framework requires also implementation of:
    
    - num_classes property: Number of classification label classes.
    - setup method: Downloads or prepares needed resources
                    For example, check if images are 
                    downloaded and if they are not, download them.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Tensor, int, Any]:
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def setup(self) -> None:
        # download data and whatnot if needed
        raise NotImplementedError
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError