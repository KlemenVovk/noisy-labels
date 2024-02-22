from typing import Any, Callable, Dict
from abc import abstractmethod, ABC

import torch
from torch import Tensor

# EARLY ALPHA STUFF; SUBJECT TO CHANGES

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

class Noise(ABC):
    """Base class for noise functions.
    _generated dict instance variable holds cached noisy labels {index: noisy_label},
    so they can be shared among different dataset instances.
    """
    _generated: Dict[int, int]

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
        return int(noisy_target)

    @abstractmethod
    def _noisify_target(self, feature: Tensor, target: int | Tensor, index: int | Tensor) -> int:
        raise NotImplementedError
    
    # TODO: could be improved in the future,
    # saving only _generated does not work
    # as you need to save fcn as well for LambdaNoise for example
    def save_state(self, fpath: str) -> "Noise":
        torch.save(self, fpath)

    @staticmethod
    def load_state(fpath: str) -> "Noise":
        return torch.load(fpath)


class InstanceNoise(Noise):
    """Generates noisy targets from a precomputed vector of noisy targets.
    """

    def __init__(self, noisy_targets: Tensor) -> None:
        super().__init__()
        # TODO: _generated can be precached here
        self.noisy_targets = noisy_targets # vector of noisy labels

    def _noisify_target(self, feature: Tensor, target: int | Tensor, index: int | Tensor) -> int:
        return self.noisy_targets[index]


class AsymmetricNoise(Noise):
    """Generates noisy labels with an arbitrary noise transtion matrix.
    """
    def __init__(self, transition_matrix: Tensor) -> None:
        super().__init__()
        self.transition_matrix = transition_matrix

    def _noisify_target(self, feature: Tensor, target: int | Tensor, index: int | Tensor) -> int:
        return torch.multinomial(self.transition_matrix[target], num_samples=1).item()
    

class SymmetricNoise(AsymmetricNoise):
    """Generates noisy targets with a symmetric noise transition matrix.
    """

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
    """Generates noisy targets using a custom function.
    The custom function must accept feature, target, index and return a noisy target.
    """
    def __init__(self, fcn: Callable[[Tensor, int, int], int]) -> None:
        """Initialises LambdaNoise object.

        Args:
            fcn (Callable): The custom function must accept feature, target, index and return a noisy target.
        """
        super().__init__()
        self.fcn = fcn

    def _noisify_target(self, feature: Tensor, target: int | Tensor, index: int | Tensor) -> int:
        return self.fcn(feature, target, index)


class BiasedSymmetricNoise(Noise):
    """
        Symmetric noise implementation based on 
        https://github.com/LiJunnan1992/DivideMix/blob/d9d3058fa69a952463b896f84730378cdee6ec39/dataloader_cifar.py#L58-L78
        This is different from our implementation where the expected noise rate is noise_rate,
        whereas here the expected noise rate is noise_rate - 1/num_classes * noise_rate.
    """
    def __init__(self, noise_rate, num_samples, num_classes) -> None:
        super().__init__()
        idx = torch.randperm(num_samples)
        num_noise = int(noise_rate*num_samples)
        self.noise_idx = idx[:num_noise]
        self.num_classes = num_classes

    def _noisify_target(self, feature: Tensor, target: int | Tensor, index: int | Tensor) -> int:
        if index in self.noise_idx:
            return torch.randint(0, self.num_classes, (1,)).item()
        return target