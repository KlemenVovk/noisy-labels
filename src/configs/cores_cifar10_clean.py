from methods.classifiers.resnet import resnet34
from methods.learning_strategies.cores2.cores2 import SampleSieve

from configs.base import MethodConfig
from configs.cifar10 import cifar10_base_config

class cores_cifar10_clean(MethodConfig):

    data_config = cifar10_base_config

    classifier = resnet34
    classifier_args = dict(
        weights=None,
        num_classes=10
    )

    learning_strategy = SampleSieve
    learning_strategy_args = dict(
        initial_lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    trainer_args = dict(
        deterministic=True,
        max_epochs=100
    )
