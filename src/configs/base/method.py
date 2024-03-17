from typing import Tuple, Type, Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from lightning import LightningModule, LightningDataModule, Trainer, seed_everything

from configs.base.base import Config
from configs.base.data import DataConfig


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

    # this is handled entirely in method's lightning 
    # module so anything really can be passed in here
    optimizer_cls: Type[Optimizer] | list[Type[Optimizer]] = None
    optimizer_args: dict | list[dict] = dict()
    scheduler_cls: Type[LRScheduler] | list[Type[LRScheduler]] = None
    scheduler_args: dict | list[dict] = dict()

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
            datamodule=datamodule,
            classifier_cls=cls.classifier, classifier_args=cls.classifier_args,
            optimizer_cls=cls.optimizer_cls, optimizer_args=cls.optimizer_args,
            scheduler_cls=cls.scheduler_cls, scheduler_args=cls.scheduler_args,
            **cls.learning_strategy_args
        )

        # trainer - needs to be initialised here because seed active needs to be run beforehand
        trainer = Trainer(**cls.trainer_args)

        return model, datamodule, trainer