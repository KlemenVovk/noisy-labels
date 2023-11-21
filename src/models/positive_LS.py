from torch.nn import CrossEntropyLoss

from models.CE import CE

# same as CE baseline but with label smoothing?

class PositiveLS(CE):
    def __init__(self, label_smoothing, initial_lr, momentum, weight_decay, datamodule):
        super().__init__(initial_lr, momentum, weight_decay, datamodule)
        self.criterion = CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing) # basic CE with label smoothing