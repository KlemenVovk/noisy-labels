from .CAL import CAL_config
from .CE import CE_config
from .co_teaching import co_teaching_config
from .co_teaching_plus import co_teaching_plus_config
from .cores2 import cores2_config
from .divide_mix import divide_mix_config
from .divide_mix_clean import divide_mix_clean_config
from .ELR import ELR_config
from .ELR_plus import ELR_plus_config
from .forward_T import forwardT_config
from .backward_T import backwardT_config
from .GCE import GCE_config
from .jocor import jocor_config
from .peer_loss import peer_loss_config
from .PES import PES_config
from .PES_semi import PES_semi_config
from .SOP import SOP_config
from .SOP_plus import SOP_plus_config
from .T_revision import TRevision_config
from .volminnet import volminnet_config
from .pro_mix import pro_mix_config

method_configs = {
    "CAL": CAL_config,
    "CE": CE_config,
    "co_teaching": co_teaching_config,
    "co_teaching_plus": co_teaching_plus_config,
    "cores2": cores2_config,
    "divide_mix": divide_mix_config,
    "divide_mix_clean": divide_mix_clean_config,
    "ELR": ELR_config,
    "ELR_plus": ELR_plus_config,
    "GCE": GCE_config,
    "jocor": jocor_config,
    "PES": PES_config,
    "PES_semi": PES_semi_config,
    "SOP": SOP_config,
    "SOP_plus": SOP_plus_config,
    "volminnet": volminnet_config,
    "pro_mix": pro_mix_config,
}

__all__ = [
    "CAL_config",
    "CE_config",
    "co_teaching_config",
    "co_teaching_plus_config",
    "cores2_config",
    "divide_mix_config",
    "divide_mix_clean_config",
    "ELR_config",
    "ELR_plus_config",
    "forwardT_config",
    "backwardT_config",
    "GCE_config",
    "jocor_config",
    "peer_loss_config",
    "PES_config",
    "PES_semi_config",
    "SOP_config",
    "SOP_plus_config",
    "TRevision_config",
    "volminnet_config",
    "pro_mix_config",
    "method_configs",
]
