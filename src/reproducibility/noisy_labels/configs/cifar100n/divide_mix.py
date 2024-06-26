from noisypy.methods.learning_strategies.divide_mix.divide_mix import DivideMix
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from ..base.utils import PreResNet18
from ..base.config import CIFAR100NoisyLabelsMethod
from ..base.wrappers import dividemixify_wrapper


class divide_mix_config(CIFAR100NoisyLabelsMethod):

    _data_config_wrapper = dividemixify_wrapper
    classifier = PreResNet18

    learning_strategy_cls = DivideMix
    learning_strategy_args = dict(
        warmup_epochs=30,
        noise_type="asymmetric",
        noise_rate=0.5,
        p_thresh=0.5,
        temperature=0.5,
        alpha=4,
        lambda_u=25,
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.02, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = LambdaLR
    scheduler_args = dict(lr_lambda=lambda epoch: 0.1 if epoch >= 150 else 1)

    trainer_args = {
        **CIFAR100NoisyLabelsMethod.trainer_args,
        "max_epochs": 300,
        "reload_dataloaders_every_n_epochs": 1,
    }
