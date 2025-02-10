from typing import Any, Type
from itertools import accumulate

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


class LearningStrategyWithWarmupModule(LearningStrategyModule):

    def __init__(self, 
                 datamodule: L.LightningDataModule, 
                 classifier_cls: type, classifier_args: dict, 
                 optimizer_cls: type[Optimizer], optimizer_args: dict, 
                 scheduler_cls: type[LRScheduler], scheduler_args: dict,
                 warmup_epochs: int,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args, optimizer_cls, 
            optimizer_args, scheduler_cls, scheduler_args, *args, **kwargs
        )
        self.warmup_epochs = warmup_epochs


class MultiStageLearningStrategyModule(LearningStrategyModule):
    
    def __init__(self, 
                 datamodule: L.LightningDataModule, 
                 classifier_cls: Type, classifier_args: dict,
                 optimizer_cls: Optimizer, optimizer_args: dict,
                 scheduler_cls: LRScheduler, scheduler_args: dict,
                 stage_epochs: list[int], # num of epochs per stage, e.g. [20, 30, 100]
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(
            datamodule, classifier_cls, classifier_args, optimizer_cls,
            optimizer_args, scheduler_cls, scheduler_args, *args, **kwargs
        )
        self.stage_epoch_cumsum = list(accumulate(stage_epochs))
        self.automatic_optimization = False
    
    @property
    def current_stage(self):
        valid = [i for i, e in enumerate(self.stage_epoch_cumsum) if self.current_epoch <= e]
        return min(valid)
    
    def _get_current_stage_method(self, method_name):
        # returns the method: method_name_stage{current_stage}
        # or a sink for compatibility
        entries = [e for e in dir(self) if method_name+"_stage" in e]
        entries = sorted(entries, key=lambda e: int(e[-1]))
        if len(entries) > self.current_stage:
            method_name = entries[self.current_stage]
            return getattr(self, method_name)
        return lambda *_: None # just a dummy sink function

    def training_step(self, batch: Any, batch_idx: int) -> None:
        # get step and optim for current stage
        training_step = self._get_current_stage_method("training_step")
        optim = self.optimizers()[self.current_stage] 
        
        # training_step + optimize
        loss = training_step(batch, batch_idx)
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

    def on_train_epoch_end(self) -> None:
        on_train_epoch_end = self._get_current_stage_method("on_train_epoch_end")
        sch = self.lr_schedulers()[self.current_stage]
        
        on_train_epoch_end()
        sch.step()
    
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        validation_step = self._get_current_stage_method("validation_step")
        validation_step(batch, batch_idx)

    def on_validation_end(self) -> None:
        on_validation_end = self._get_current_stage_method("on_validation_end")
        on_validation_end()