from typing import List

from .basic import BasicDataModule
from .multi_sample import MultiSampleDataModule, to_list
from .registry import DATAMODULES
from ..noise.noises import Noise
from ..noise.dataset_wrapper import noisify_dataset

# TODO: this is getting out of control
#       think about making some sort of pipeline/factory
#       that takes list of datamodule wrappers
#       and applies them in sequence
#       so you can avoid defining each datamodule from 0

@DATAMODULES.register_module("noise-multisample")
class MulitSampleDataModuleNoisify(MultiSampleDataModule):

    def __init__(self, 
                 noise: Noise,
                 train_dataset_cls, *args, **kwargs) -> None:
        # add noise to train dataset
        train_dataset_cls = [
            noisify_dataset(td, noise) for td in to_list(train_dataset_cls)]
        super().__init__(*args, **kwargs, train_dataset_cls=train_dataset_cls)
    
@DATAMODULES.register_module("noise")
class MulitSampleDataModuleNoisify(BasicDataModule):

    def __init__(self, 
                 noise: Noise,
                 train_dataset_cls, *args, **kwargs) -> None:
        # add noise to train dataset
        train_dataset_cls = noisify_dataset(train_dataset_cls, noise)
        super().__init__(*args, **kwargs, train_dataset_cls=train_dataset_cls)