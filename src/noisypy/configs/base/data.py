from typing import Any, Callable, Type

from lightning import LightningDataModule

from noisypy.data.datasets.base import DatasetFW
from noisypy.data.datamodule import MultiSampleDataModule
from noisypy.data.pipelines.base import AugmentationPipeline, Identity
from .base import Config


class DataConfig(Config):
    """Data configuration. Holds all classes and arguments needed to generate a datamodule for a specific method.
    To configure a new data pipeline, inherit from this class and change the needed class variables.

    Class vars:
        dataset_cls (Type[DatasetFW]):  Class of dataset.
        dataset_args: (dict) :          Dict of keyword args to initialise the dataset_cls. Can be incomplete and will be updated by dataset_{train/val/test}_args class var.

        dataset_train_augmentation (AugmentationPipeline | list[AugmentationPipeline]): Augmentation pipeline to transform dataset_cls for train. If list is provided, multiple samples will be returned at each step.
        dataset_val_augmentation:  (AugmentationPipeline | list[AugmentationPipeline]): Augmentation pipeline to transform dataset_cls for val. If list is provided, multiple samples will be returned at each step.
        dataset_test_augmentation: (AugmentationPipeline | list[AugmentationPipeline]): Augmentation pipeline to transform dataset_cls for test. If list is provided, multiple samples will be returned at each step.

        num_train_samples (int): Number of samples of train dataset for each train step, only supported if a single augmentation is provided. Raises AssertionError if dataset_train_augmentation is a list.
        num_val_samples   (int): Number of samples of val dataset for each val step, only supported if a single augmentation is provided. Raises AssertionError if dataset_val_augmentation is a list.
        num_test_samples  (int): Number of samples of test dataset for each test step, only supported if a single augmentation is provided. Raises AssertionError if dataset_test_augmentation is a list.

        dataset_train_args (dict | list[dict]): Dict or list of dicts of keyword arguments to be added to dataset_args class var. If list is provided, must be broadcastable with number of samples.
        dataset_val_args (dict | list[dict]): Dict or list of dicts of keyword arguments to be added to dataset_args class var. If list is provided, must be broadcastable with number of samples.
        dataset_test_args (dict | list[dict]): Dict or list of dicts of keyword arguments to be added to dataset_args class var. If list is provided, must be broadcastable with number of samples.

        datamodule_cls (Type[MultiSampleDataModule]): Class of datamodule.
        datamodule_args (dict): Dict of remaining keyword arguments to initialise datamodule_cls, that are not train {train/val/test}_dataset_{cls/args}.
    """

    # classes
    dataset_train_cls: Type[DatasetFW] = None
    dataset_val_cls: Type[DatasetFW] | None = None
    dataset_test_cls: Type[DatasetFW] | None = None

    # common arguments
    dataset_args: dict = dict()

    # augmentation pipelines for subsets
    dataset_train_augmentation: AugmentationPipeline | list[AugmentationPipeline] = (
        Identity()
    )
    dataset_val_augmentation: AugmentationPipeline | list[AugmentationPipeline] = (
        Identity()
    )
    dataset_test_augmentation: AugmentationPipeline | list[AugmentationPipeline] = (
        Identity()
    )

    # changes to dataset_args for subset
    dataset_train_args: dict | list[dict] = dict()
    dataset_val_args: dict | list[dict] = dict()
    dataset_test_args: dict | list[dict] = dict()

    # datamodule class and args that are not {train/test/val}_dataset_{cls/args}
    datamodule_args: dict = dict()

    @classmethod
    def build_modules(cls) -> LightningDataModule:
        # apply augmentations
        train_dataset_cls = apply_augmentations(
            cls.dataset_train_cls, cls.dataset_train_augmentation
        )
        val_dataset_cls = (
            apply_augmentations(cls.dataset_val_cls, cls.dataset_val_augmentation)
            if cls.dataset_val_cls
            else []
        )
        test_dataset_cls = (
            apply_augmentations(cls.dataset_test_cls, cls.dataset_test_augmentation)
            if cls.dataset_test_cls
            else []
        )

        # update kwargs for each subset
        train_args = merge_args(cls.dataset_args, cls.dataset_train_args)
        val_args = merge_args(cls.dataset_args, cls.dataset_val_args)
        test_args = merge_args(cls.dataset_args, cls.dataset_test_args)

        # init subset datasets
        train_datasets = broadcast_init(train_dataset_cls, train_args)
        val_datasets = (
            broadcast_init(val_dataset_cls, val_args) if val_dataset_cls else []
        )
        test_datasets = (
            broadcast_init(test_dataset_cls, test_args) if test_dataset_cls else []
        )

        # init datamodule
        datamodule = MultiSampleDataModule(
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            test_datasets=test_datasets,
            **cls.datamodule_args,
        )
        return datamodule


def ensure_list(obj: list | Any) -> list[Any]:
    return obj if isinstance(obj, list) else [obj]


def is_broadcastable(obj1: Any, obj2: Any) -> bool:
    obj1, obj2 = ensure_list(obj1), ensure_list(obj2)
    return len(obj1) == 1 or len(obj2) == 1 or len(obj1) == len(obj2)


def broadcast_init(
    classes: Callable[[Any], object] | list[Callable[[Any], object]],
    kwss: dict | list[dict],
) -> list[object]:
    assert is_broadcastable(
        classes, kwss
    ), f"Classes {classes} and kwargs {kwss} are not broadcastable."
    classes, kwss = ensure_list(classes), ensure_list(kwss)
    if len(classes) == 1:
        if len(kwss) == 1:
            return [classes[0](**kwss[0])]  # 1, 1
        return [classes[0](**kws) for kws in kwss]  # 1, n
    if len(classes) == len(kwss):
        return [class_(**kws) for class_, kws in zip(classes, kwss)]  # n, n
    return [class_(**kwss[0]) for class_ in classes]  # 1, n


def apply_augmentations(
    dataset_cls: Type[DatasetFW],
    augmentations: AugmentationPipeline | list[AugmentationPipeline],
) -> list[Type[DatasetFW]]:
    augmentations = ensure_list(augmentations)
    return [aug(dataset_cls) for aug in augmentations]


def merge_args(base_args: dict, update_args: dict | list[dict]) -> list[dict]:
    update_args = ensure_list(update_args)
    return [{**base_args, **update} for update in update_args]
