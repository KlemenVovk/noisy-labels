from typing import Any


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
        class UpdatedConfig(cls):
            pass

        setattr(UpdatedConfig, field_name, new_value)
        return UpdatedConfig
