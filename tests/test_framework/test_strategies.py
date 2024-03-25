import pytest
from typing import Type
from noisypy.configs.base.method import MethodConfig
from .perf_targets_reprod import performance_targets as reprod_targets
from .perf_targets_framework import performance_targets as fw_targets
from .utils import Target, PerfTargetCheckCallback


@pytest.mark.parametrize("config, target_dict", [(k, v) for k, v in reprod_targets.items()])
def test_reprod_perf(config: Type[MethodConfig], target_dict: dict[str, tuple[Target]], capsys):
    with capsys.disabled(): # disable stdout capturing - can be removed when the test works
        module, datamodule, trainer = config.build_modules()
        trainer.callbacks.append(PerfTargetCheckCallback(target_dict))
        trainer.fit(model=module, datamodule=datamodule)

    for targets in target_dict.values():
        for target in targets:
            assert target.hit, f"{target} @ epoch {target.epoch} was hit with {target.hit_value}"


@pytest.mark.parametrize("config, target_dict", [(k, v) for k, v in fw_targets.items()])
def test_framework_perf(config: Type[MethodConfig], target_dict: dict[str, tuple[Target]], capsys):
    with capsys.disabled(): # disable stdout capturing - can be removed when the test works
        module, datamodule, trainer = config.build_modules()
        trainer.callbacks.pop(-1) # remove checkpointing
        trainer.callbacks.append(PerfTargetCheckCallback(target_dict))
        trainer.fit(model=module, datamodule=datamodule)

    for targets in target_dict.values():
        for target in targets:
            assert target.hit, f"{target} @ epoch {target.epoch} was hit with {target.hit_value}"