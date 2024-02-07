import lightning as L


class Target:

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch
        self.hit_value = None
        self.hit = False

    def shoot(self, epoch: int, value: float) -> None:
        if epoch == self.epoch:
            self.hit_value = value
            self.hit = self.check_hit(value)
    
    def check_hit(self, value: float) -> bool:
        raise NotImplementedError
    
    def __str__(self) -> str:
        return f"Target(epoch={self.epoch})"

    def __repr__(self) -> str:
        return str(self)
    
class RangeTarget(Target):

    def __init__(self, epoch: int, min: float, max: float) -> None:
        super().__init__(epoch)
        self.min, self.max = min, max
    
    def check_hit(self, value: float) -> bool:
        return self.min <= value and value <= self.max

    def __str__(self) -> str:
        return super().__str__()[:-1] + f", min={self.min}, max={self.max})"
    
class ExactTarget(Target):

    def __init__(self, epoch: int, value: float, tol=1e-4) -> None:
        super().__init__(epoch)
        self.value, self.tol = value, tol

    def check_hit(self, value: float) -> bool:
        return abs(self.value - value) <= self.tol
    
    def __str__(self) -> str:
        return super().__str__()[:-1] + f", value={self.value}, tol={self.tol})"
 

class PerfTargetCheckCallback(L.Callback):

    def __init__(self, targets: list[Target] | tuple[Target]) -> None:
        super().__init__()
        self.targets = targets
        self.max_epoch = max(t.epoch for t in targets)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logs = trainer.callback_metrics
        epoch = trainer.current_epoch
        for target in self.targets:
            target.shoot(epoch, logs["train_acc"]) # TODO handle different metric targets
        if epoch == self.max_epoch:
            trainer.should_stop = True