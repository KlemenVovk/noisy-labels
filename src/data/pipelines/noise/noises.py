from typing import Any, Callable
from abc import abstractmethod

import torch
from torch import Tensor

# TODO: looking at this now, perhaps, it would be better
# if you just apply noise to whole vector of targets
# instead of on individual labels
# PROS: + perhaps faster (almost certainly if you do this in prepare())
#       + persistence of ys between batches is handled in dataset wrapper, 
#         as the noise only gets called once on all targets
# CONS: - wrapper becomes a bit dirtier, since you first need
#         to sample all the ys of original dataset, and the only
#         way to do so, is to iterate through the original dataset
#         with __getitem__(i) (or [i])

class Noise:

    def __init__(self) -> None:
        # to save and mark generated noisy labels
        # dict instead of 2 arrays
        # because it doesn't require knowing dataset size in advance
        self._generated = dict()

    def __call__(self, feature: Tensor, target: int | Tensor, index: int | Tensor) -> Any:
        # apply noise and save at index if
        # noise was not yet applied
        noisy_target = self._generated.get(index, None)
        if noisy_target is None:
            noisy_target = self._noisify_target(feature, target, index)
            self._generated[index] = noisy_target
        return noisy_target

    @abstractmethod
    def _noisify_target(self, feature: Tensor, target: int | Tensor, index: int | Tensor) -> int:
        raise NotImplementedError
    
    @classmethod
    def from_file(cls, fpath: str) -> "Noise":
        #TODO
        pass
    
    @classmethod
    def from_vector(cls, noisy_targets: Tensor) -> "Noise":
        #TODO
        pass


class InstanceNoise(Noise):

    def __init__(self, noisy_targets: Tensor) -> None:
        super().__init__()
        self.noisy_targets = noisy_targets # vector of noisy labels

    def _noisify_target(self, feature: Tensor, target: int | Tensor, index: int | Tensor) -> int:
        return self.noisy_targets[index]


class AsymmetricNoise(Noise):

    def __init__(self, transition_matrix: Tensor) -> None:
        super().__init__()
        self.transition_matrix = transition_matrix

    def _noisify_target(self, feature: Tensor, target: int | Tensor, index: int | Tensor) -> int:
        return torch.multinomial(self.transition_matrix[target], num_samples=1).item()
    

class SymmetricNoise(AsymmetricNoise):

    def __init__(self, num_classes: int, noise_rate: float) -> None:
        transition_matrix = self.generate_transition_matrix(num_classes, noise_rate)
        super().__init__(transition_matrix)

    @staticmethod
    def generate_transition_matrix(num_classes: int, noise_rate: float) -> Tensor:
        # transition matrix with (1 - noise_rate) on diagonal
        # and noise_rate / (num_classes - 1) elsewhere
        eps = noise_rate / (num_classes - 1)
        T = eps * torch.ones(num_classes, num_classes)
        T += (1 - eps - noise_rate) * torch.eye(num_classes)
        return T
    

class LambdaNoise(Noise):

    def __init__(self, fcn: Callable) -> None:
        super().__init__()
        self.fcn = fcn

    def _noisify_target(self, feature: Tensor, target: int | Tensor, index: int | Tensor) -> int:
        return self.fcn(feature, target, index)
