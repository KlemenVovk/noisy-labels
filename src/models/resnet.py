from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.models.resnet import resnet34
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch

# TODO: evaluation metrics
# TODO: aim logger
# TODO: Cores loss

class ResNet34(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet34(pretrained=True)
        self.num_classes = num_classes
        # Replace the last layer with a new one with num_classes outputs
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
