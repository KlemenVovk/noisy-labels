from typing import Type, List
from abc import abstractmethod, ABC

from ..datasets.base import DatasetFW


class AugmentationPipeline(ABC):
    # gets Dataset cls as an input and returns transformed Dataset cls 

    def __call__(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:
        return self.transform(dataset_cls)

    @abstractmethod
    def transform(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:
        raise NotImplementedError
    

class ComposePipeline(AugmentationPipeline):

    def __init__(self, augmentations: List[AugmentationPipeline]) -> None:
        super().__init__()
        self.augmentations = augmentations

    def transform(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:
        for aug in self.augmentations:
            dataset_cls = aug(dataset_cls)
        return dataset_cls
    
    
class IdentityPipeline(AugmentationPipeline):

    def transform(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:
        return dataset_cls