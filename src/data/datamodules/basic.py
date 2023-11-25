from typing import List

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from torch.utils.data import DataLoader

from .registry import DATAMODULES

# TODO
# datamodule takes: - dataset class + args
#                  (- noise class + args)
#                  (- ...)

@DATAMODULES.register_module("basic")
class BasicDataModule(LightningDataModule):

    def __init__(self, 
                 train_dataset_cls, train_dataset_kws: dict,
                 val_dataset_cls, val_dataset_kws: dict,
                 test_dataset_cls, test_dataset_kws: dict,
                 batch_size, num_workers,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.train_dataset = train_dataset_cls(**train_dataset_kws)
        self.val_dataset = val_dataset_cls(**val_dataset_kws)
        self.test_dataset = test_dataset_cls(**test_dataset_kws)

        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_train_samples(self):
        return len(self.train_dataset)
    
    @property
    def num_val_samples(self):
        return len(self.val_dataset)
    
    @property
    def num_test_samples(self):
        return len(self.test_dataset)
    
    @property
    def num_classes(self):
        return self.train_dataset.num_classes

    def setup(self, stage: str = None) -> None:
        # setup/prepare/download datasets
        self.train_dataset.setup()
        self.val_dataset.setup()
        self.test_dataset.setup()
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size, num_workers=self.num_workers)