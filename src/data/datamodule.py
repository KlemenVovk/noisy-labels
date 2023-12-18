#TODO inherit from basic and just wrap each train dataset with noise

from typing import List, Callable, Any, Type

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from torch.utils.data import DataLoader
from data.datasets.base import DatasetFW

def ensure_list(obj: list | Any) -> List[Any]:
    """Helper function that promotes objects to list if not already list

    Args:
        obj (list | Any): Object to promote. Can be instance of list, in which case it is returned as is.

    Returns:
        List[Any]: Promoted list.
    """
    return obj if isinstance(obj, list) else [obj]

def broadcastable(obj1: Any, obj2: Any) -> bool:
    """Helper function that checks if two objects are broadcastable.
    Object must have implemented __len__ method.
    Objects are broadcastable when their lenghts are: (1, 1) or (1, n) or (n, 1) or (n, n).
    Objects are not broadcastable when their lenghts are: (m, n).

    Args:
        obj1 (_type_): First object.
        obj2 (_type_): Second object.

    Returns:
        bool: Objects broadcastable(True) or not(False).
    """
    obj1, obj2 = ensure_list(obj1), ensure_list(obj2)
    return len(obj1) == 1 or\
        len(obj2) == 1 or\
        len(obj1) == len(obj2)

def broadcast_init(classes: Callable[[Any], object] | List[Callable[[Any], object]], kwss: dict | List[dict]) -> List[object]:
    """Helper function that initialises classes with the provided keyword arguments.
    Classes and arguments must be broadcastable. If classes or dicts are not an instance of list, they are promoted with ensure_list().

    Args:
        classes (Callable[[Any], object] | List[Callable[[Any], object]]): Callable or list of callables that initialise an object from kwss.
        kwss (dict | List[dict]): Dict or list of dicts of keyword arguments used to initialise classes.

    Returns:
        List[object]: List of initialised object (even if a single object was initalised).
    """
    classes, kwss = ensure_list(classes), ensure_list(kwss)
    if len(classes) == 1:
        if len(kwss) == 1: 
            return [classes[0](**kwss[0])] # 1, 1
        return [classes[0](**kws) for kws in kwss] # 1, n
    
    elif len(classes) == len(kwss):
        return [class_(**kws) for class_, kws in zip(classes, kwss)] # n, n

    return [class_(**kwss[0]) for class_ in classes] # 1, n 


class MultiSampleDataModule(LightningDataModule):
    """Lightning datamodule that supports multisampling for each of the train/val/test subests.
    """

    def __init__(self, 
                 train_dataset_cls: Type[DatasetFW] | List[Type[DatasetFW]], train_dataset_args: dict | List[dict],
                 val_dataset_cls: Type[DatasetFW] | List[Type[DatasetFW]],   val_dataset_args: dict | List[dict],
                 test_dataset_cls: Type[DatasetFW] | List[Type[DatasetFW]],  test_dataset_args: dict | List[dict],
                 batch_size: int, num_workers: int,
                 *args, **kwargs) -> None:
        """Initialises MultiSampleDataModule object.

        Args:
            train_dataset_cls (Type[DatasetFW] | List[Type[DatasetFW]]):    Class or list of classes of train datasets.
            train_dataset_args (dict | List[dict]):                         Dict or list of dicts of keyword arguments to initialise train dataset(s).
            val_dataset_cls (Type[DatasetFW] | List[Type[DatasetFW]]):      Class or list of classes of val datasets.
            val_dataset_args (dict | List[dict]):                           Dict or list of dicts of keyword arguments to initialise val dataset(s).
            test_dataset_cls (Type[DatasetFW] | List[Type[DatasetFW]]):     Class or list of classes of test datasets.
            test_dataset_args (dict | List[dict]):                          Dict or list of dicts of keyword arguments to initialise test dataset(s).
            batch_size (int): Size of batches returned by dataloaders. 
            num_workers (int): Number of cores used for sampling with the dataloaders.
        """
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
    def num_train_samples(self) -> int:
        return len(self.train_datasets[0]) # TODO: bug if different lengths - will this ever happen?
    
    @property
    def num_val_samples(self) -> int:
        return len(self.val_datasets[0]) # --||--
    
    @property
    def num_test_samples(self) -> int:
        return len(self.test_datasets[0]) # --||--
    
    @property
    def num_classes(self) -> int:
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