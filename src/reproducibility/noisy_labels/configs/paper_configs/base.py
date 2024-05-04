from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from lightning.pytorch.loggers import CSVLogger
from noisypy.configs.base.method import MethodConfig
from ..base.utils import ResNet34

class BasePaperMethod(MethodConfig):

    _data_config_wrapper = None

    classifier=ResNet34
    classifier_args=dict(
        num_classes=10,
    )

    optimizer_cls=SGD
    optimizer_args=dict(
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler_cls=MultiStepLR
    scheduler_args=dict(
        milestones=[60],
        gamma=0.1
    )

    trainer_args = dict(
        max_epochs=100,
        deterministic=True,
        logger=CSVLogger("../../logs/paper_configs", name="NONE"),
    )

    seed = 1337