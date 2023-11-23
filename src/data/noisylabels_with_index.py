# CLEANUP TODO: it would be best if Noisylabels datasets 
# had a parameter whether to return index or not
# and Datamodule was passed Dataset as a parameter
# only problem is data prepare phase if non-cifar datset is used
# if thats the case, add .prepare() to dataset and call that maybe?

from .noisylabels import NoisyLabelsCIFAR10Dataset, NoisyLabelsCIFAR100Dataset, NoisylabelsDataModule, CIFAR10, CIFAR100

class NoisylabelsCIFAR10WithIndexDataset(NoisyLabelsCIFAR10Dataset): # ik lol
    
    def __getitem__(self, index):
        return *super().__getitem__(index), index
    
class NoisylabelsCIFAR100WithIndexDataset(NoisyLabelsCIFAR100Dataset): # ik h
    
    def __getitem__(self, index):
        return *super().__getitem__(index), index
    
class NoisylabelsWithIndexDataModule(NoisylabelsDataModule): # ikr lol

    def setup(self, stage: str = None) -> None:
        if self.dataset_name == "cifar10":
            self.train_dataset = NoisylabelsCIFAR10WithIndexDataset(self.noise_type, self.noisylabels_dir, self.cifar_dir, False, transform=self.train_transform)
            self.test_dataset = CIFAR10(self.cifar_dir, train=False, transform=self.test_transform)
        else:
            self.train_dataset = NoisylabelsCIFAR100WithIndexDataset(self.noise_type, self.noisylabels_dir, self.cifar_dir, False, transform=self.train_transform)
            self.test_dataset = CIFAR100(self.cifar_dir, train=False, transform=self.test_transform)