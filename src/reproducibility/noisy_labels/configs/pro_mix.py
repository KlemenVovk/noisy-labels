from noisypy.methods.learning_strategies.pro_mix.pro_mix import ProMix
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base.utils import PreResNet18
from .base.config import NoisyLabelsMethod
from .base.wrappers import promixify_wrapper


class pro_mix_config(NoisyLabelsMethod):
    _data_config_wrapper = promixify_wrapper
    classifier = PreResNet18

    learning_strategy_cls = ProMix
    learning_strategy_args = dict(
        warmup_epochs=10,
        rampup_epochs=50,
        noise_type="symmetric",
        rho_start=0.5,
        rho_end=0.5,
        debias_beta_pl=0.8,
        alpha_output=0.8,
        tau=0.99,
        start_expand=250,
        threshold=0.9,
        bias_m=0.9999,
        temperature=0.5,
        model_type="paper_resnet",
        feat_dim=128,
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = CosineAnnealingLR
    scheduler_args = dict(T_max=600, eta_min=5e-5)

    trainer_args = {
        **NoisyLabelsMethod.trainer_args,
        "max_epochs": 600,
    }
