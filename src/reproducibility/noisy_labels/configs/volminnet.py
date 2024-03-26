from noisypy.methods.learning_strategies.volminnet.volminnet import VolMinNet
from .base.config import NoisyLabelsMethod


class volminnet_config(NoisyLabelsMethod):

    learning_strategy_cls = VolMinNet
    learning_strategy_args = dict(
        lam=1e-4
    )

    optimizer_cls = [NoisyLabelsMethod.optimizer_cls]*2
    optimizer_args = [NoisyLabelsMethod.optimizer_args]*2

    scheduler_cls = [NoisyLabelsMethod.scheduler_cls]*2
    scheduler_args = [NoisyLabelsMethod.scheduler_args]*2
