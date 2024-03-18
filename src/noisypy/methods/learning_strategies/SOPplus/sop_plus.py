from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch.nn.functional import one_hot

from ..SOP.sop import SOP


class SOPplus(SOP):           
    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x1, x2, y_noise, index = batch[0]
        opt, opt_loss = self.optimizers()
        self.model.train()
        
        y = one_hot(y_noise, 10).float() # TODO: if 10 is num of classes - change to self.datamodule.num_classes so it's not hard-coded
        x_all = torch.cat([x1, x2])

        y_pred = self.model(x_all)
        loss = self.criterion(index, y_pred, y)
        self.train_acc(y_pred, torch.cat([y_noise, y_noise]))

        # optimization step
        opt_loss.zero_grad()
        opt.zero_grad()
        loss.backward()
        opt_loss.step()
        opt.step()
        # logging
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        return loss