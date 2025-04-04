from noisypy.methods.learning_strategies.DISC.DISC import DISC
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from ..base import BenchmarkConfigCIFAR10N
from ...base.wrappers import disc_aug_wrapper


class DISC_config(BenchmarkConfigCIFAR10N):

    _data_config_wrapper = disc_aug_wrapper

    learning_strategy_cls = DISC
    learning_strategy_args = dict(
        start_epoch=15, # https://github.com/JackYFL/DISC/blob/a21e910bffeb34873684937ac1066991a720552a/algorithms/DISC.py#L81
        alpha=5.0, 
        sigma=0.5,
        momentum=0.99,
        lambd_ce=1.0,
        lambd_h=1.0, # 0.2 for larger noise ratios
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0, weight_decay=0.001)
    scheduler_cls = MultiStepLR
    scheduler_args = dict(milestones=[80, 160])
    
    trainer_args = {
        **BenchmarkConfigCIFAR10N.trainer_args,
        "reload_dataloaders_every_n_epochs": 1,
        "max_epochs": 200,
    }
