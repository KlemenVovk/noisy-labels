from abc import abstractmethod, ABC
from typing import List, Dict, Any, Tuple, Type, Callable

from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from aim.pytorch_lightning import AimLogger

from data.datasets.base import DatasetFW
from data.datamodule import MultiSampleDataModule, ensure_list
from data.pipelines.base import AugmentationPipeline, IdentityPipeline

#TODO think about: when you inherit from base config and want to change
# some_args dict, you would want to only update the dict
# which can be done by _merge_args(super().some_args, new_args), which is not the cleanest thing
# there are probably some libraries for configs that are more intuitive than our approach
# so maybe it's better to look at other options also
#TODO add support for different loggers in MethodConfig

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
    """

    dataset_cls: Type[DatasetFW] = None
    dataset_args: Dict = dict()

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
    datamodule_cls: type = MultiSampleDataModule
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
    """
    
    # data pipeline configuration used for generating datamodule
    data_config: DataConfig = DataConfig()

    # function or class and needed args to initalize a classifier
    classifier: Callable  = None
    classifier_args: dict = dict()

    # module of the strategy and additional parameters that are not classifier, or datamodule
    learning_strategy_cls: Type[LightningModule] = None
    learning_strategy_args: dict = dict()

    # lightning trainer and additional parameters that are not logger
    trainer: Type[Trainer] = Trainer
    trainer_args: dict = dict(deterministic=True)

    # god seed
    seed: int = 1337

    @classmethod
    def build_modules(cls) -> Tuple[LightningModule, LightningDataModule, Trainer]:
        # seed
        seed_everything(cls.seed, workers=True)

        # datamodule
        datamodule = cls.data_config.build_modules()

        # lightning module
        model = cls.learning_strategy_cls(
            classifier_cls=cls.classifier, classifier_args=cls.classifier_args,
            datamodule=datamodule, **cls.learning_strategy_args
        )

        # TODO: support for different loggers
        # logger
        aim_logger = AimLogger(
            experiment=cls.learning_strategy_cls.__name__,
            train_metric_prefix='train_',
            test_metric_prefix='test_',
            val_metric_prefix='val_',
        )

        # trainer
        trainer = cls.trainer(
            logger=aim_logger,
            **cls.trainer_args
        )

        return model, datamodule, trainer