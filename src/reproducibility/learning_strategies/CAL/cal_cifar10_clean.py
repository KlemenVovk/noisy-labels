from pytorch_lightning.loggers import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from .utils import resnet_cifar34
from noisypy.methods.learning_strategies.CAL.cal import CAL
from noisypy.methods.learning_strategies.CAL.utils import SegAlpha
from noisypy.data.pipelines.index import AddIndex
from noisypy.configs.base.method import MethodConfig
from noisypy.configs.data.cifar10 import cifar10_base_config

class cifar10_index_config(cifar10_base_config):

    dataset_train_augmentation = AddIndex()

class cal_cifar10_clean(MethodConfig):

    data_config = cifar10_index_config

    classifier = resnet_cifar34
    classifier_args = dict(
        num_classes = 10
    )

    learning_strategy_cls = CAL
    learning_strategy_args = dict(alpha = 0.0, 
                                  warmup_epochs = 65,
                                  alpha_scheduler_cls = SegAlpha,
                                  alpha_scheduler_args = dict(
                                    alpha_list = [0.0, 1.0, 1.0],
                                    milestones = [10, 40, 80],
                                  ),
                                  alpha_scheduler_args_warmup = dict(
                                    alpha_list = [0.0, 2.0],
                                    milestones = [10, 40],
                                  ),
    ) 

    optimizer_cls = SGD
    optimizer_args = dict(lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler_cls = StepLR
    scheduler_args = dict(step_size = 60, gamma=0.1)

    trainer_args = dict(
        max_epochs=165,
        deterministic=True,
        # TODO: move to logs when we are not running from src/
        logger=CSVLogger("../logs", name="cal_clean")
    )

    seed = 1337