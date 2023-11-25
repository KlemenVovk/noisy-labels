from typing import List


class MasterRegister:

    def __init__(self) -> None:
        self.registries = {}

    def add_registry(self, registry: "Registry") -> None:
        self.registries[registry.name] = registry

    def __getitem__(self, key):
        return self.registries[key]

REGISTER = MasterRegister()


class Registry:

    def __init__(self, name: str) -> None:
        self.name = name
        self._modules = {}
        REGISTER.add_registry(self)
    
    def register_module(self, name: str) -> type:
        def decorator(stuff):
            self._modules[name] = stuff
            return stuff
        return decorator
    
    @property
    def modules(self) -> dict:
        return self._modules
    
    def __getitem__(self, key):
        return self._modules[key]