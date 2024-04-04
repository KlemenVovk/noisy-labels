from noisypy.methods.learning_strategies.CAL.cal import CAL
from noisypy.methods.learning_strategies.CAL.utils import SegAlpha
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from .base.config import NoisyLabelsMethod
from .base.wrappers import add_index_wrapper


class CAL_config(NoisyLabelsMethod):

    _data_config_wrapper = add_index_wrapper

    learning_strategy_cls = CAL
    learning_strategy_args = dict(
        alpha = 0.0, 
        warmup_epochs = 65,
        alpha_scheduler_cls = SegAlpha,
        alpha_scheduler_args = dict(
            alpha_list = [0.0, 1.0, 1.0],
            milestones = [10, 40, 80],
        ),
        alpha_scheduler_args_warmup = dict(
            alpha_list = [0.0, 2.0],
            milestones = [10, 40],
        ),
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler_cls = StepLR
    scheduler_args = dict(step_size = 60, gamma=0.1)