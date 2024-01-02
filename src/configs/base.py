from abc import abstractmethod, ABC
from typing import List, Any, Tuple, Type, Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger, CSVLogger # typing

from data.datasets.base import DatasetFW
from data.datamodule import MultiSampleDataModule, ensure_list
from data.pipelines.base import AugmentationPipeline, IdentityPipeline

# TODO I think I finally realised what's bothering me about this config system.
# The thing is that though it is better than doing yaml files, the hierarchy
# still doesn't make sense. You have to know how the modules are initialised
# in the background to effectively write a config with no errors.
# For a normal user this will not be the case. Think about how this could be improved.

def _apply_augmentations(
        dataset_cls: Type[DatasetFW], 
        augmentations: AugmentationPipeline | List[AugmentationPipeline], 
        num_samples: int = 1
        ) -> List[Type[DatasetFW]]:
    """Helper function for applying augmentations to dataset_cls in batches.

    Args:
        dataset_cls (Type[DatasetFW]): Dataset class on which the augmentations will be applied.
        augmentations (AugmentationPipeline | List[AugmentationPipeline]): Augmentation pipeline or list of augmentation pipelines to apply over a dataset.
        num_samples (int, optional): Number of samples if only a single Augmentation pipeline is provided. Defaults to 1.

    Returns:
        List[Type[DatasetFW]]: List of augmented datasets (even if a single augmentation was applied).
    """
    if num_samples != 1:
        assert isinstance(augmentations, AugmentationPipeline),\
            "Multiple samples only supported for a single augmentation."
    if not isinstance(augmentations, list):
        augmentations = [augmentations for _ in range(num_samples)]
    return [aug(dataset_cls) for aug in augmentations]


def _merge_args(base_args: dict, update_args: dict | List[dict]) -> List[dict]:
    """Helper function for merging args in batches.

    Args:
        base_args (dict): Base arguments dict to update/extend.
        update_args (dict | List[dict]): List or a single argument dict to update the base_args.

    Returns:
        List[dict]: List of updated dicts (even if a single update_args dict was provided).
    """
    update_args = ensure_list(update_args)
    return [{**base_args, **update} for update in update_args]


class Config(ABC):

    @classmethod
    @abstractmethod
    def build_modules(cls) -> Any:
        raise NotImplementedError


class DataConfig(Config):
    """Data configuration. Holds all classes and arguments needed to generate a datamodule for a specific method.
    To configure a new data pipeline, inherit from this class and change the needed class variables.

    Class vars:
        dataset_cls (Type[DatasetFW]):  Class of dataset. 
        dataset_args: (dict) :          Dict of keyword args to initialise the dataset_cls. Can be incomplete and will be updated by dataset_{train/val/test}_args class var.

        dataset_train_augmentation (AugmentationPipeline | List[AugmentationPipeline]): Augmentation pipeline to transform dataset_cls for train. If list is provided, multiple samples will be returned at each step.
        dataset_val_augmentation:  (AugmentationPipeline | List[AugmentationPipeline]): Augmentation pipeline to transform dataset_cls for val. If list is provided, multiple samples will be returned at each step.
        dataset_test_augmentation: (AugmentationPipeline | List[AugmentationPipeline]): Augmentation pipeline to transform dataset_cls for test. If list is provided, multiple samples will be returned at each step.

        num_train_samples (int): Number of samples of train dataset for each train step, only supported if a single augmentation is provided. Raises AssertionError if dataset_train_augmentation is a list.
        num_val_samples   (int): Number of samples of val dataset for each val step, only supported if a single augmentation is provided. Raises AssertionError if dataset_val_augmentation is a list.
        num_test_samples  (int): Number of samples of test dataset for each test step, only supported if a single augmentation is provided. Raises AssertionError if dataset_test_augmentation is a list.

        dataset_train_args (dict | List[dict]): Dict or list of dicts of keyword arguments to be added to dataset_args class var. If list is provided, must be broadcastable with number of samples.
        dataset_val_args (dict | List[dict]): Dict or list of dicts of keyword arguments to be added to dataset_args class var. If list is provided, must be broadcastable with number of samples.
        dataset_test_args (dict | List[dict]): Dict or list of dicts of keyword arguments to be added to dataset_args class var. If list is provided, must be broadcastable with number of samples.

        datamodule_cls (Type[MultiSampleDataModule]): Class of datamodule.
        datamodule_args (dict): Dict of remaining keyword arguments to initialise datamodule_cls, that are not train {train/val/test}_dataset_{cls/args}.
    """

    dataset_cls: Type[DatasetFW] = None
    dataset_args: dict = dict()

    # augmentation pipelines for subsets
    dataset_train_augmentation: AugmentationPipeline | List[AugmentationPipeline] = IdentityPipeline()
    dataset_val_augmentation:   AugmentationPipeline | List[AugmentationPipeline] = IdentityPipeline()
    dataset_test_augmentation:  AugmentationPipeline | List[AugmentationPipeline] = IdentityPipeline()

    # number of samples for each subset
    num_train_samples: int = 1
    num_val_samples:   int = 1
    num_test_samples:  int = 1

    # changes to dataset_args for subset
    dataset_train_args: dict | List[dict] = dict()
    dataset_val_args:   dict | List[dict] = dict()
    dataset_test_args:  dict | List[dict] = dict()

    # datamodule class and args that are not {train/test/val}_dataset_{cls/args}
    datamodule_cls: Type[MultiSampleDataModule] = MultiSampleDataModule
    datamodule_args: dict = dict()

    @classmethod
    def build_modules(cls) -> LightningDataModule:
        # datasets
        train_dataset_cls = _apply_augmentations(
            cls.dataset_cls, cls.dataset_train_augmentation, cls.num_train_samples)
        val_dataset_cls =   _apply_augmentations(
            cls.dataset_cls, cls.dataset_val_augmentation, cls.num_val_samples)
        test_dataset_cls =  _apply_augmentations(
            cls.dataset_cls, cls.dataset_test_augmentation, cls.num_test_samples)

        # dataset args
        train_args = _merge_args(cls.dataset_args, cls.dataset_train_args)
        val_args   = _merge_args(cls.dataset_args, cls.dataset_val_args)
        test_args  = _merge_args(cls.dataset_args, cls.dataset_test_args)

        # datamodule
        datamodule = cls.datamodule_cls(
            train_dataset_cls=train_dataset_cls, train_dataset_args=train_args,
            val_dataset_cls=val_dataset_cls,     val_dataset_args=val_args,
            test_dataset_cls=test_dataset_cls,   test_dataset_args=test_args,
            **cls.datamodule_args
        )
        return datamodule


class MethodConfig(Config):
    """Method configuration. Holds all classes and arguments
    needed to generate datamodule, lightningmodule and trainer for a specific method.
    To configure a new method configuration, inherit from this class and change the needed class variables.

    Class vars:
        data_config: Configuration object for data. See DataConfig class.

        classifier (Callable): Class of function to initialise a classifier torch model.
        classifier_args (dict): Dict of keyword arguments passed to classifier.

        learning_strategy_cls (Type[LightningModule]): Class of LightningModule learning strategy.
        learning_strategy_args (dict): Dict of keyword arguments to initialise learning_strategy_cls, that are not classifier_{cls/args} or datamodule.

        trainer_args (dict): Dict of keyword arguments to initialise trainer.
    """
    
    # data pipeline configuration used for generating datamodule
    data_config: DataConfig = DataConfig()

    # function or class and needed args to initalize a classifier
    classifier: Callable  = None
    classifier_args: dict = dict()

    # module of the strategy and additional parameters that are not classifier, or datamodule
    learning_strategy_cls: Type[LightningModule] = None
    learning_strategy_args: dict = dict()

    optimizer_cls: Type[Optimizer] = None
    optimizer_args: dict = dict()
    scheduler_cls: Type[LRScheduler] = None
    scheduler_args: dict = dict()

    # lightning trainer and additional parameters that are not logger
    trainer_args: dict = dict()

    # god seed
    seed: int | None = None

    @classmethod
    def build_modules(cls) -> Tuple[LightningModule, LightningDataModule, Trainer]:
        # seed
        if cls.seed is not None:
            seed_everything(cls.seed, workers=True)

        # datamodule
        datamodule = cls.data_config.build_modules()

        # lightning module
        model = cls.learning_strategy_cls(
            classifier_cls=cls.classifier, classifier_args=cls.classifier_args,
            datamodule=datamodule,
            optimizer_cls=cls.optimizer_cls, optimizer_args=cls.optimizer_args,
            scheduler_cls=cls.scheduler_cls, scheduler_args=cls.scheduler_args,
            **cls.learning_strategy_args
        )

        # trainer - needs to be initialised here because seed active needs to be run beforehand
        trainer = Trainer(**cls.trainer_args)

        return model, datamodule, trainer