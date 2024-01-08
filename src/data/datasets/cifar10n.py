from typing import Literal, Callable, Any
from .noisylabels import NoisyLabelsLoader
from .cifar10 import CIFAR10


class CIFAR10N(CIFAR10):

    def __init__(
            self, 
            noise_type: Literal[
                "clean_label", "aggre_label", "worse_label",
                "random_label1", "random_label2", "random_label3"
                ],
            noise_dir: str,
            cifar_dir: str,
            train: bool = True,
            transform: Callable[..., Any] | None = None,
            target_transform: Callable[..., Any] | None = None,
            download: bool = False,
            ) -> None:
        super().__init__(cifar_dir, train, transform, target_transform, download)
        self.noisy_targets = NoisyLabelsLoader("cifar10", noise_dir, download).load_label(noise_type)
        
    def __getitem__(self, index):
        img, true_target = super().__getitem__(index)
        if not self.train: # test set has no labels
            return img, true_target
        return img, int(self.noisy_targets[index])