from noisypy.configs.data.cifar10n import (cifar10n_clean_config, 
                                           cifar10n_aggre_config,
                                           cifar10n_worse_config,
                                           cifar10n_random1_config,
                                           cifar10n_random2_config,
                                           cifar10n_random3_config)
from .CE_cifar10_clean import CE_cifar10_clean


class CE_cifar10n_clean(CE_cifar10_clean):

    data_config = cifar10n_clean_config


class CE_cifar10n_aggre(CE_cifar10_clean):

    data_config = cifar10n_aggre_config


class CE_cifar10n_worse(CE_cifar10_clean):

    data_config = cifar10n_worse_config


class CE_cifar10n_random1(CE_cifar10_clean):

    data_config = cifar10n_random1_config


class CE_cifar10n_random2(CE_cifar10_clean):

    data_config = cifar10n_random2_config


class CE_cifar10n_random3(CE_cifar10_clean):

    data_config = cifar10n_random3_config
