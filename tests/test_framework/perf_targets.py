import pytest
from typing import Type

from src.configs.base import Config
from src.configs.methods.cores2.cores_cifar10_clean import cores_cifar10_clean

class Target:

    def __init__(
        self, 
        min: float,
        max: float,
        exact: float | None = None,
        tol: float = 1e-3) -> None:
        self.min, self.max, self.exact, self.tol = min, max, exact, tol
        
    def check_hit(self, value: float) -> bool:
        return self.min <= value <= self.max
    
    def check_hit_exact(self, value: float) -> bool:
        return value == self.exact


# {config class: {epoch: target}}
performance_targets: dict[Type[Config], dict[int, Target]] = {
    cores_cifar10_clean: {
        10: Target(68, 76, 73.7420),
        20: Target(75, 80)
    }
}


