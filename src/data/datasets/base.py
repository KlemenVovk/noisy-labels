from abc import abstractmethod, ABC
from typing import Any

from torch.utils.data.dataset import Dataset as DatasetPT


class DatasetFW(DatasetPT, ABC):

    @abstractmethod
    def __getitem__(self, index) -> Any:
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