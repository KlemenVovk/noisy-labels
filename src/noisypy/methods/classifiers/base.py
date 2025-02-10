from typing import Any, Protocol

# A callable that takes num classes as input argument so we can autofill it during config building

class Classifier(Protocol):

    def __call__(self, num_classes: int, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
