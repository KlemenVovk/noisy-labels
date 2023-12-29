from typing import Any, Type

import lightning as L

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LearningStrategyModule(L.LightningModule):
    
    def __init__(self, 
                 datamodule: L.LightningDataModule,
                 classifier_cls: type, classifier_args: dict,
                 optimizer_cls: Type[Optimizer], optimizer_args: dict,
                 scheduler_cls: Type[LRScheduler], scheduler_args: dict,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.datamodule = datamodule
        self.classifier_cls, self.classifier_args = classifier_cls, classifier_args
        self.optimizer_cls, self.optimizer_args = optimizer_cls, optimizer_args
        self.scheduler_cls, self.scheduler_args = scheduler_cls, scheduler_args
