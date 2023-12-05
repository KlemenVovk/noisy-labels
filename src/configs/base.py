from abc import abstractmethod
from typing import List, Union, Dict, Any, Tuple

from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from aim.pytorch_lightning import AimLogger

from data.datasets.base import Dataset
from data.datamodule import MultiSampleDataModule
from data.pipelines.base import AugmentationPipeline, IdentityPipeline

#TODO typehints
#TODO add support for train, test and val augmentations
#TODO add support for multiple samples in DataConfig
#TODO add support for different loggers in MethodConfig

class Config:

    @classmethod
    @abstractmethod
    def build_modules() -> Any:
        raise NotImplementedError


class DataConfig(Config):

    dataset_cls: type = None
    dataset_args: Dict = dict()
    dataset_train_augmentation: AugmentationPipeline = IdentityPipeline()
    dataset_val_augmentation:   AugmentationPipeline = IdentityPipeline()
    dataset_test_augmentation:  AugmentationPipeline = IdentityPipeline()
    dataset_train_args: Dict = dict()
    dataset_val_args: Dict = dict()
    dataset_test_args: Dict = dict()

    datamodule_cls: type = MultiSampleDataModule
    datamodule_args: Dict = dict()

    @classmethod
    def build_modules(cls) -> LightningDataModule:
        # TODO add support for multiple samples
        # datasets
        train_dataset_cls = cls.dataset_train_augmentation(cls.dataset_cls)
        val_dataset_cls =   cls.dataset_test_augmentation(cls.dataset_cls)
        test_dataset_cls =  cls.dataset_val_augmentation(cls.dataset_cls)

        # dataset args
        train_args = {**cls.dataset_args, **cls.dataset_train_args}
        val_args   = {**cls.dataset_args, **cls.dataset_val_args}
        test_args  = {**cls.dataset_args, **cls.dataset_test_args}

        # datamodule
        datamodule = cls.datamodule_cls(
            train_dataset_cls=train_dataset_cls, train_dataset_args=train_args,
            val_dataset_cls=val_dataset_cls,     val_dataset_args=val_args,
            test_dataset_cls=test_dataset_cls,   test_dataset_args=test_args,
            **cls.datamodule_args
        )
        return datamodule


class MethodConfig(Config):
    
    data_config: DataConfig = DataConfig()

    classifier = None
    classifier_args = dict()

    learning_strategy = None
    learning_strategy_args = dict()

    trainer = Trainer
    trainer_args = dict(deterministic=True)

    seed = 1337

    @classmethod
    def build_modules(cls) -> Tuple[LightningModule, LightningDataModule, Trainer]:
        # seed
        seed_everything(cls.seed, workers=True)

        # datamodule
        datamodule = cls.data_config.build_modules()

        # lightning module
        model = cls.learning_strategy(
            classifier_cls=cls.classifier, classifier_args=cls.classifier_args,
            datamodule=datamodule, **cls.learning_strategy_args
        )

        # TODO: support for different loggers
        # logger
        aim_logger = AimLogger(
            experiment=cls.learning_strategy.__name__,
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