import pytest
from typing import Type
from configs.base import MethodConfig
from .perf_targets import performance_targets
from .utils import Target, PerfTargetCheckCallback

@pytest.mark.parametrize("config,targets", [(k, v) for k, v in performance_targets.items()])
def test_performance(config: Type[MethodConfig], targets: tuple[Target], capsys):
    with capsys.disabled():
        module, datamodule, trainer = config.build_modules()
        trainer.callbacks.append(PerfTargetCheckCallback(targets))
        trainer.fit(model=module, datamodule=datamodule)

    for target in targets:
        assert target.hit, f"{target} @ epoch {target.epoch} was hit with {target.hit_value}"