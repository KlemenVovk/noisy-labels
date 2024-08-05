from typing import Literal, Callable, Any
from .noisylabels import NoisyLabelsLoader
from .cifar100 import CIFAR100


class CIFAR100N(CIFAR100):

    def __init__(
            self, 
            noise_type: Literal[
                "clean_label", "noisy_label",
                "noisy_coarse_label", "clean_coarse_label",
                ],
            noise_dir: str,
            cifar_dir: str,
            train: bool = True,
            transform: Callable[..., Any] | None = None,
            target_transform: Callable[..., Any] | None = None,
            download: bool = False,
            ) -> None:
        super().__init__(cifar_dir, train, transform, target_transform, download)
        self.noisy_targets = NoisyLabelsLoader("cifar100", noise_dir, download).load_label(noise_type)
        
    def __getitem__(self, index):
        img, true_target = super().__getitem__(index)
        if not self.train: # test set has no labels
            return img, true_target
        return img, int(self.noisy_targets[index])