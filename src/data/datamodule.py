#TODO inherit from basic and just wrap each train dataset with noise

from typing import List, Callable

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from torch.utils.data import DataLoader

# broadcastable: seq lenghts: 1, 1 ok; 1, n ok; n, 1 ok; n, n ok; n, m err

def to_list(obj):
    return obj if isinstance(obj, list) else [obj]

def broadcastable(obj1, obj2):
    obj1, obj2 = to_list(obj1), to_list(obj2)
    return len(obj1) == 1 or\
        len(obj2) == 1 or\
        len(obj1) == len(obj2)

def broadcast_init(classes: Callable | List[Callable], kwss: dict | List[dict]) -> List[object]:
    # assume that classes and kws have are broadcastable
    classes, kwss = to_list(classes), to_list(kwss)
    if len(classes) == 1:
        if len(kwss) == 1: 
            return [classes[0](**kwss[0])] # 1, 1
        return [classes[0](**kws) for kws in kwss] # 1, n
    
    elif len(classes) == len(kwss):
        return [class_(**kws) for class_, kws in zip(classes, kwss)] # n, n

    return [class_(**kwss[0]) for class_ in classes] # 1, n 


class MultiSampleDataModule(LightningDataModule):

    def __init__(self, 
                 train_dataset_cls, train_dataset_args: dict | List[dict],
                 val_dataset_cls, val_dataset_args: dict | List[dict],
                 test_dataset_cls, test_dataset_args: dict | List[dict],
                 batch_size, num_workers,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        err_msg = lambda type, dataset, args: f"Mismatched {type} dataset cls and args lengths: {len(dataset)}, {len(args)}"
        assert broadcastable(train_dataset_cls, train_dataset_args), err_msg("train", train_dataset_cls, train_dataset_args)
        assert broadcastable(val_dataset_cls,   val_dataset_args),   err_msg("val",   val_dataset_cls,   val_dataset_args)
        assert broadcastable(test_dataset_cls,  test_dataset_args),  err_msg("test",  test_dataset_cls,  test_dataset_args)

        self.train_datasets = broadcast_init(train_dataset_cls, train_dataset_args)
        self.val_datasets   = broadcast_init(val_dataset_cls,   val_dataset_args)
        self.test_datasets  = broadcast_init(test_dataset_cls,  test_dataset_args)

        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_train_samples(self):
        return len(self.train_datasets[0]) # TODO: bug if different lengths - will this ever happen?
    
    @property
    def num_val_samples(self):
        return len(self.val_datasets[0]) # --||--
    
    @property
    def num_test_samples(self):
        return len(self.test_datasets[0]) # --||--
    
    @property
    def num_classes(self):
        return self.train_datasets[0].num_classes

    def setup(self, stage: str = None) -> None:
        # setup/prepare/download datasets
        for train_dataset in self.train_datasets:
            train_dataset.setup()
        for val_dataset in self.val_datasets:
            val_dataset.setup()
        for test_dataset in self.test_datasets:
            test_dataset.setup()
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return [DataLoader(td, self.batch_size, shuffle=True, num_workers=self.num_workers)
                for td in self.train_datasets]
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(vd, self.batch_size, num_workers=self.num_workers)
                for vd in self.val_datasets]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(td, self.batch_size, num_workers=self.num_workers)
                for td in self.test_datasets]