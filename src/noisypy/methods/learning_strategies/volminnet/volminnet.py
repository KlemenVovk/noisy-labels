from typing import Any, Type

import lightning as L
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torchmetrics

from ..base import LearningStrategyModule
from .utils import SigT


class VolMinNet(LearningStrategyModule):

    def __init__(
        self,
        datamodule: L.LightningDataModule,
        classifier_cls: type,
        classifier_args: dict,
        optimizer_cls: Type[Optimizer],
        optimizer_args: dict,
        scheduler_cls: Type[LRScheduler],
        scheduler_args: dict,
        lam: float,
        init_t: float,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            datamodule,
            classifier_cls,
            classifier_args,
            optimizer_cls,
            optimizer_args,
            scheduler_cls,
            scheduler_args,
            *args,
            **kwargs,
        )
        self.save_hyperparameters("lam")

        self.model = classifier_cls(**classifier_args)
        self.t_model = SigT(self.datamodule.num_classes, init=init_t)
        self.criterion = (
            lambda logits, true, T: F.nll_loss(logits, true)
            + lam * T.slogdet().logabsdet
        )

        # metrics
        N = datamodule.num_classes
        self.train_acc = torchmetrics.Accuracy(
            num_classes=N, top_k=1, task="multiclass", average="micro"
        )
        self.val_acc = torchmetrics.Accuracy(
            num_classes=N, top_k=1, task="multiclass", average="micro"
        )
        self.test_acc = torchmetrics.Accuracy(
            num_classes=N, top_k=1, task="multiclass", average="micro"
        )

        self.automatic_optimization = False

    def _predict(self, x):
        logits = self.model(x)
        T = self.t_model()
        corrected_logits = (F.softmax(logits, dim=-1) @ T).log()
        return corrected_logits, T

    def training_step(self, batch: Any, batch_idx: int) -> None:
        optim, optim_T = self.optimizers()

        # forward
        x, y = batch[0]
        logits, T = self._predict(x)
        loss = self.criterion(logits, y, T)

        # backward + step
        optim.zero_grad()
        optim_T.zero_grad()
        self.manual_backward(loss)
        optim.step()
        optim_T.step()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(logits, y), on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self) -> None:
        sch, sch_T = self.lr_schedulers()
        sch.step()
        sch_T.step()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits, T = self._predict(x)
        loss = self.criterion(logits, y, T)

        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(logits, y))

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_pred, _ = self._predict(x)
        self.test_acc(y_pred, y)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def configure_optimizers(self):
        optim = self.optimizer_cls[0](self.model.parameters(), **self.optimizer_args[0])
        optim_T = self.optimizer_cls[1](
            self.t_model.parameters(), **self.optimizer_args[1]
        )

        sch = self.scheduler_cls[0](optim, **self.scheduler_args[0])
        sch_T = self.scheduler_cls[1](optim_T, **self.scheduler_args[1])
        return [optim, optim_T], [sch, sch_T]
