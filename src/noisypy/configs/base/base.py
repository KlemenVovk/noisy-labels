from typing import Any
from copy import deepcopy

# TODO I think I finally realised what's bothering me about this config system.
# The thing is that though it is better than doing yaml files, the hierarchy
# still doesn't make sense. You have to know how the modules are initialised
# in the background to effectively write a config with no errors.
# For a normal user this will not be the case. Think about how this could be improved.
# ANSWER: METAPROGRAMMING BABY: when you hit a dict argument update it with current dict ;D

class ConfigMeta(type):

    def __new__(cls, clsname, bases, attrs):
        base_attrs = {}
        for base in bases:
            base_attrs = {**base_attrs, **vars(base)}
        return super().__new__(cls, clsname, bases, attrs)


class Config(metaclass=ConfigMeta):

    @classmethod
    def build_modules(cls) -> Any:
        raise NotImplementedError

    @classmethod
    def update_field(cls, field_name, new_value) -> "Config":
        class UpdatedConfig(cls): pass
        setattr(UpdatedConfig, field_name, new_value)
        return UpdatedConfig
