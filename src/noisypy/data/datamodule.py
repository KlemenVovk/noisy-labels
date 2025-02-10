from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from torch.utils.data import DataLoader
from .datasets.base import DatasetFW


class MultiSampleDataModule(LightningDataModule):
    """Lightning datamodule that supports multiple samples for each of the train/val/test subests.
    """

    def __init__(self, 
                 train_datasets: list[DatasetFW], 
                 val_datasets: list[DatasetFW],
                 test_datasets: list[DatasetFW],
                 batch_size: int, 
                 num_workers: int,
                 train_shuffle: bool = True, val_shuffle: bool = False, test_shuffle: bool = False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_datasets = train_datasets
        self.val_datasets   = val_datasets
        self.test_datasets  = test_datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle

    @property
    def num_train_samples(self) -> int:
        return len(self.train_datasets[0])
    
    @property
    def num_val_samples(self) -> int:
        return len(self.val_datasets[0]) if self.val_datasets else 0
    
    @property
    def num_test_samples(self) -> int:
        return len(self.test_datasets[0]) if self.test_datasets else 0
    
    @property
    def num_classes(self) -> int:
        return self.train_datasets[0].num_classes

    def setup(self, stage: str = None) -> None:
        pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return [DataLoader(td, self.batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers)
                for td in self.train_datasets]
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(vd, self.batch_size, shuffle=self.val_shuffle, num_workers=self.num_workers)
                for vd in self.val_datasets]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(td, self.batch_size, shuffle=self.test_shuffle, num_workers=self.num_workers)
                for td in self.test_datasets]