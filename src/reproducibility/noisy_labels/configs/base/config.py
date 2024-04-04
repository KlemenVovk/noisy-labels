from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from lightning.pytorch.loggers import CSVLogger
from noisypy.configs.base.method import MethodConfig
from .utils import ResNet34

class NoisyLabelsMethod(MethodConfig):

    _data_config_wrapper = None

    classifier=ResNet34
    classifier_args=dict(
        num_classes=10,
    )

    trainer_args = dict(
        max_epochs=1,
        deterministic=True,
        logger=CSVLogger("../logs", name="NONE"),
    )

    seed = 1337