from typing import Type
from configs.base import MethodConfig
from configs.methods.cores2.cores_cifar10_clean import cores_cifar10_clean

from .utils import Target, RangeTarget, ExactTarget

# currently hard coded to train_acc
# TODO: handle other metrics somehow
performance_targets: dict[Type[MethodConfig], dict[str, tuple[Target]]] = {
    cores_cifar10_clean: {
        "train_acc": (
            RangeTarget(epoch=10, min=0.68, max=0.76),
            ExactTarget(epoch=10, value=0.737420, tol=1e-4),
        ),
    }
}