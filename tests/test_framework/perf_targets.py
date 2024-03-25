from typing import Type
from noisypy.configs.base.method import MethodConfig
from .utils import Target, RangeTarget, ExactTarget

from noisypy.configs.methods.CE.CE_cifar10_clean import CE_cifar10_clean_config
from reproducibility.learning_strategies.co_teaching.co_teaching_reprod import co_teaching_reprod
from reproducibility.learning_strategies.co_teaching.co_teaching_plus_reprod import co_teaching_plus_reprod
from reproducibility.learning_strategies.divide_mix.divide_mix_cifar10_clean import divide_mix_cifar10_clean
from reproducibility.learning_strategies.ELR.elr_cifar10_clean import elr_cifar10_clean
from reproducibility.learning_strategies.ELRplus.elr_plus_cifar10_clean import elr_plus_cifar10_clean
from reproducibility.learning_strategies.cores2.cores_cifar10_clean import cores_cifar10_clean
from reproducibility.learning_strategies.jocor.jocor_reprod import jocor_reprod
from reproducibility.learning_strategies.peer_loss.peer_loss_reprod import peer_loss_reprod
from reproducibility.learning_strategies.PES.pes_cifar10_noise import pes_cifar10_noise
from reproducibility.learning_strategies.PESsemi.pes_semi_cifar10_clean import pes_semi_cifar10n_clean
from reproducibility.learning_strategies.SOP.sop_cifar10_clean import sop_cifar10_clean
from reproducibility.learning_strategies.t_revision.t_revision_reprod import t_revision_reprod
from reproducibility.learning_strategies.volminnet.volminnet_reprod import volminnet_reprod


performance_targets: dict[Type[MethodConfig], dict[str, tuple[Target]]] = {

    CE_cifar10_clean_config: {
        "train_acc": (
            ExactTarget(epoch=0, value=0.174500003, tol=1e-4),
        ),
    },

    co_teaching_reprod: {
        "train_acc1": (
            ExactTarget(epoch=0, value=0.3701399863, tol=1e-4),
        ),
    },

    co_teaching_plus_reprod: {
        "train_acc1": (
            ExactTarget(epoch=0, value=0.2563599944, tol=1e-4),
        ),
    },
    
    cores_cifar10_clean: {
        "train_acc": (
            ExactTarget(epoch=0, value=0.3554799855, tol=1e-4),
            #ExactTarget(epoch=10, value=0.737420, tol=1e-4),
        ),
    },

    divide_mix_cifar10_clean: {
        "val_acc": (
            ExactTarget(epoch=0, value=0.6452000, tol=1e-4),
        ),
    },

    elr_cifar10_clean: {
        "train_acc": (
            ExactTarget(epoch=0, value=0.39912000, tol=1e-4),
        ),
    },

    elr_plus_cifar10_clean: {
        "val_acc": (
            ExactTarget(epoch=0, value=0.558300018, tol=1e-4),
        ),
    },

    jocor_reprod: {
        "train_acc1": (
            ExactTarget(epoch=0, value=0.2775599957, tol=1e-4),
        ),
    },

    peer_loss_reprod: {
        "train_acc": (
            ExactTarget(epoch=0, value=0.351839, tol=1e-4),
        ),
    },

    pes_cifar10_noise: {
        "train_acc": (
            ExactTarget(epoch=0, value=0.1371400058, tol=1e-4),
        ),
    },

    t_revision_reprod: {
        "train_acc": (
            ExactTarget(epoch=0, value=0.3147599995, tol=1e-4),
        ),
    },

    volminnet_reprod: {
        "train_acc": (
            ExactTarget(epoch=0, value=0.3339999914, tol=1e-4),
        ),
    },
    
}