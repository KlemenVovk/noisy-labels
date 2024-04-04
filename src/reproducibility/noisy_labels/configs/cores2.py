from noisypy.methods.learning_strategies.cores2.cores2 import SampleSieve
from noisypy.methods.learning_strategies.cores2.utils import f_beta
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from .base.config import NoisyLabelsMethod


lr_plan = [0.1] * 50 + [0.01] * (50 + 1)

class cores2_config(NoisyLabelsMethod):

    learning_strategy_cls = SampleSieve

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_cls = LambdaLR
    scheduler_args = dict(lr_lambda=lambda epoch: lr_plan[epoch] / (1+f_beta(epoch)))