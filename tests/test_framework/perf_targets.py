from typing import Type
from configs.base.method import MethodConfig
from .utils import Target, RangeTarget, ExactTarget

from configs.methods.CE.CE_cifar10_clean import CE_cifar10_clean
from configs.methods.co_teaching.co_teaching_reprod import co_teaching_reprod
from configs.methods.cores2.cores_cifar10_clean import cores_cifar10_clean
from configs.methods.volminnet.volminnet_reprod import volminnet_reprod
from configs.methods.jocor.jocor_reprod import jocor_reprod


performance_targets: dict[Type[MethodConfig], dict[str, tuple[Target]]] = {

    CE_cifar10_clean: {
        "train_acc": (
            ExactTarget(epoch=0, value=0.1972000003, tol=1e-4),
        ),
    },

    co_teaching_reprod: {
        "train_acc1": (
            ExactTarget(epoch=0, value=0.3701399863, tol=1e-4),
        ),
    },
    
    cores_cifar10_clean: {
        "train_acc": (
            ExactTarget(epoch=0, value=0.3554799855, tol=1e-4),
            #ExactTarget(epoch=10, value=0.737420, tol=1e-4),
        ),
    },

    volminnet_reprod: {
        "train_acc": (
            ExactTarget(epoch=0, value=0.3339999914, tol=1e-4),
        ),
    },

    jocor_reprod: {
        "train_acc1": (
            ExactTarget(epoch=0, value=0.2775599957, tol=1e-4),
        ),
    },
    
}