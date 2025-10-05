from lightning.pytorch.loggers import CSVLogger
from noisypy.configs.base.method import MethodConfig
from torchvision.models import resnet34


class BenchmarkConfigCIFAR10N(MethodConfig):
    _data_config_wrapper = None

    classifier = resnet34
    classifier_args = dict(
        num_classes=10,
        weights=None,
    )

    trainer_args = dict(
        max_epochs=1,
        deterministic=True,
        enable_checkpointing=False,
        logger=CSVLogger("../../logs/benchmark", name="NONE"),
    )

    seed = 1337


class BenchmarkConfigCIFAR100N(BenchmarkConfigCIFAR10N):
    classifier_args = dict(
        num_classes=100,
        weights=None,
    )
