from typing import Type, List
from abc import abstractmethod, ABC

from ..datasets.base import DatasetFW


class AugmentationPipeline(ABC):
    """Base class for dataset augmentation pipelines.
    Dataset augmentation pipeline should transform() DatasetFW class and return DatasetFW class
    """

    def __call__(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:
        """Applies transform() to dataset class.

        Args:
            dataset_cls (Type[DatasetFW]): Dataset class to transform.

        Returns:
            Type[DatasetFW]: Transformed dataset class.
        """
        return self.transform(dataset_cls)

    @abstractmethod
    def transform(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:
        raise NotImplementedError
    

class ComposePipeline(AugmentationPipeline):
    """Composition of different dataset augmentation pipelines.
    Applies each augmentation sequentially.
    """

    def __init__(self, augmentations: List[AugmentationPipeline]) -> None:
        """Initialises ComposePipeline object.

        Args:
            augmentations (List[AugmentationPipeline]): Sequence of augmentations to apply to dataset class.
        """
        super().__init__()
        self.augmentations = augmentations

    def transform(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:
        for aug in self.augmentations:
            dataset_cls = aug(dataset_cls)
        return dataset_cls
    
    
class IdentityPipeline(AugmentationPipeline):
    """Identity transformation of a dataset class (no changes applied).
    Useful for placeholders in default configs.
    """

    def transform(self, dataset_cls: Type[DatasetFW]) -> Type[DatasetFW]:
        return dataset_cls