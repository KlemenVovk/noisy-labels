from .CAL import CAL_config
from .CE import CE_config
from .co_teaching import co_teaching_config
from .co_teaching_plus import co_teaching_plus_config
from .divide_mix import divide_mix_config
from .divide_mix_clean import divide_mix_clean_config
from .ELR import ELR_config
from .ELR_plus import ELR_plus_config
from .PES_semi import PES_semi_config
from .SOP import SOP_config
from .SOP_plus import SOP_plus_config
from .volminnet import volminnet_config

method_configs = {
    "CAL": CAL_config,
    "CE": CE_config,
    "co_teaching": co_teaching_config,
    "co_teaching_plus": co_teaching_plus_config,
    "divide_mix": divide_mix_config,
    "divide_mix_clean": divide_mix_clean_config,
    "ELR": ELR_config,
    "ELR_plus": ELR_plus_config,
    "PES_semi": PES_semi_config,
    "SOP": SOP_config,
    "SOP_plus": SOP_plus_config,
    "volminnet": volminnet_config,
}

__all__ = [
    "CAL_config",
    "CE_config",
    "co_teaching_config",
    "co_teaching_plus_config",
    "divide_mix_config",
    "divide_mix_clean_config",
    "ELR_config",
    "ELR_plus_config",
    "PES_semi_config",
    "SOP_config",
    "SOP_plus_config",
    "volminnet_config",
    "method_configs",
]
