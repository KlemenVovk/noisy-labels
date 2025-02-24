from typing import Type
from noisypy.configs.base.method import MethodConfig
from .utils import Target, ExactTarget

from noisypy.configs.methods.CAL.CAL_cifar10_clean import CAL_cifar10_clean_config
from noisypy.configs.methods.CE.CE_cifar10_clean import CE_cifar10_clean_config
from noisypy.configs.methods.co_teaching.co_teaching_cifar10_clean import (
    co_teaching_cifar10_clean_config,
)
from noisypy.configs.methods.co_teaching.co_teaching_plus_cifar10_clean import (
    co_teaching_plus_cifar10_clean_config,
)
from noisypy.configs.methods.cores2.cores2_cifar10_clean import (
    cores2_cifar10_clean_config,
)
from noisypy.configs.methods.divide_mix.divide_mix_cifar10_clean import (
    divide_mix_cifar10_clean_config,
)
from noisypy.configs.methods.ELR.ELR_cifar10_clean import ELR_cifar10_clean_config
from noisypy.configs.methods.ELRplus.ELRplus_cifar10_clean import (
    ELRplus_cifar10_clean_config,
)
from noisypy.configs.methods.FBT.backwardT_cifar10_clean import (
    backwardT_cifar10_clean_config,
)
from noisypy.configs.methods.FBT.forwardT_cifar10_clean import (
    forwardT_cifar10_clean_config,
)
from noisypy.configs.methods.GCE.GCE_cifar10_clean import GCE_cifar10_clean_config
from noisypy.configs.methods.jocor.jocor_cifar10_clean import jocor_cifar10_clean_config
from noisypy.configs.methods.peer_loss.peer_loss_cifar10_clean import (
    peer_loss_cifar10_clean_config,
)
from noisypy.configs.methods.PES.PES_cifar10_clean import PES_cifar10_clean_config
from noisypy.configs.methods.PESsemi.PESsemi_cifar10_clean import (
    PESsemi_cifar10n_clean_config,
)
from noisypy.configs.methods.SOP.SOP_cifar10_clean import SOP_cifar10_clean_config
from noisypy.configs.methods.SOPplus.SOPplus_cifar10_clean import (
    SOPplus_cifar10_clean_config,
)
from noisypy.configs.methods.t_revision.t_revision_cifar10_clean import (
    t_revision_cifar10_clean_config,
)
from noisypy.configs.methods.volminnet.volminnet_cifar10_clean import (
    volminnet_cifar10_clean_config,
)


performance_targets: dict[Type[MethodConfig], dict[str, tuple[Target]]] = {
    CAL_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.16147500276565552, tol=1e-4),),
    },
    CE_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.1745000034570694, tol=1e-4),),
    },
    co_teaching_cifar10_clean_config: {
        "train_acc1": (ExactTarget(epoch=0, value=0.38727501034736633, tol=1e-4),),
    },
    co_teaching_plus_cifar10_clean_config: {
        "train_acc1": (ExactTarget(epoch=0, value=0.38727501034736633, tol=1e-4),),
    },
    cores2_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.3350749909877777, tol=1e-4),),
    },
    divide_mix_cifar10_clean_config: {
        "val_acc": (ExactTarget(epoch=0, value=0.46239998936653137, tol=1e-4),),
    },
    ELR_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.2971999943256378, tol=1e-4),),
    },
    ELRplus_cifar10_clean_config: {
        "val_acc": (ExactTarget(epoch=0, value=0.5333999991416931, tol=1e-4),),
    },
    backwardT_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.1745000034570694, tol=1e-4),),
    },
    forwardT_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.1745000034570694, tol=1e-4),),
    },
    GCE_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.3169499933719635, tol=1e-4),),
    },
    jocor_cifar10_clean_config: {
        "train_acc1": (ExactTarget(epoch=0, value=0.23340000212192535, tol=1e-4),),
    },
    peer_loss_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.2983500063419342, tol=1e-4),),
    },
    PES_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.19292500615119934, tol=1e-4),),
    },
    PESsemi_cifar10n_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.3105500042438507, tol=1e-4),),
    },
    SOP_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.3249250054359436, tol=1e-4),),
    },
    SOPplus_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.24718749523162842, tol=1e-4),),
    },
    t_revision_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.2827250063419342, tol=1e-4),),
    },
    volminnet_cifar10_clean_config: {
        "train_acc": (ExactTarget(epoch=0, value=0.33377501368522644, tol=1e-4),),
    },
}
