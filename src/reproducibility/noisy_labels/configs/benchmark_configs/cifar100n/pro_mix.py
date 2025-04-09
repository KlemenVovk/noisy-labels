from noisypy.methods.learning_strategies.pro_mix.pro_mix import ProMix
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..base import BenchmarkConfigCIFAR100N
from ...base.wrappers import promixify_wrapper

class pro_mix_config(BenchmarkConfigCIFAR100N):
    _data_config_wrapper = promixify_wrapper

    learning_strategy_cls = ProMix
    # https://github.com/Justherozen/ProMix/blob/40d8378193f1a098473479ebcb56dbafa89dfacb/run.sh#L21C140-L21C153
    learning_strategy_args = dict(
        warmup_epochs=30,
        rampup_epochs=50,
        noise_type="symmetric",
        rho_start=0.5,
        rho_end=0.5,
        debias_beta_pl=0.5,
        alpha_output=0.5,
        tau=0.95,
        start_expand=250,
        threshold=0.9,
        bias_m=0.9999,
        temperature=0.5,
        model_type="pytorch_resnet",
        feat_dim=128,
    )

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = CosineAnnealingLR
    scheduler_args = dict(T_max=600, eta_min=5e-5)

    trainer_args = {
        **BenchmarkConfigCIFAR100N.trainer_args,
        "max_epochs": 600,
    }
