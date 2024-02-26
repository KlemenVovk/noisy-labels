from abc import abstractmethod, ABC
from typing import Any

# TODO I think I finally realised what's bothering me about this config system.
# The thing is that though it is better than doing yaml files, the hierarchy
# still doesn't make sense. You have to know how the modules are initialised
# in the background to effectively write a config with no errors.
# For a normal user this will not be the case. Think about how this could be improved.


class Config(ABC):

    @classmethod
    @abstractmethod
    def build_modules(cls) -> Any:
        raise NotImplementedError


