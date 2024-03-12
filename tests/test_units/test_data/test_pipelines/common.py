from typing import Any, Tuple
from noisypy.data.datasets.cifar10 import CIFAR10

class CIFAR10WithExtras(CIFAR10):
    """Adds 2 extra outputs in getitem.
    "something" and "something_else" are appended to each sample.
    For testing purposes only
    """

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return (*super().__getitem__(index), "something", "something_else")


args = dict(root="data/cifar", train=True)