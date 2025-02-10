from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from noisypy.methods.learning_strategies.peer_loss.utils import lr_plan
from noisypy.methods.learning_strategies.peer_loss.peer_loss import PeerLoss

from .base import BasePaperMethod

# First stage is warmup, as per https://openreview.net/pdf?id=TBWA6PLJZQm E.3
# and communication with authors using 200 epochs with CE.
stages = [200, 50] 
class peer_loss_config(BasePaperMethod):

    learning_strategy_cls = PeerLoss
    learning_strategy_args = dict(
        stage_epochs=stages,
        warmup_criterion=CrossEntropyLoss,
    )

    optimizer_cls = [BasePaperMethod.optimizer_cls, BasePaperMethod.optimizer_cls]
    optimizer_args = [
        BasePaperMethod.optimizer_args,
        BasePaperMethod.optimizer_args,
    ]
    
    scheduler_cls = [BasePaperMethod.scheduler_cls, BasePaperMethod.scheduler_cls]
    scheduler_args = [
        BasePaperMethod.scheduler_args,
        BasePaperMethod.scheduler_args,
    ]

    trainer_args = {
        **BasePaperMethod.trainer_args,
        "max_epochs": sum(stages)+1,
    }