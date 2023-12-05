from typing import TypeVar, List
from abc import abstractmethod

from ..datasets.base import Dataset

#TODO typing

class AugmentationPipeline:

    def __call__(self, dataset_cls):
        return self.transform(dataset_cls)

    @abstractmethod
    def transform(self, dataset_cls):
        raise NotImplementedError
    

class ComposePipeline(AugmentationPipeline):

    def __init__(self, augmentations: List[Dataset]) -> None:
        super().__init__()
        self.augmentations = augmentations

    def transform(self, dataset_cls):
        for aug in self.augmentations:
            dataset_cls = aug(dataset_cls)
        return dataset_cls
    
    
class IdentityPipeline(AugmentationPipeline):

    def transform(self, dataset_cls):
        return dataset_cls