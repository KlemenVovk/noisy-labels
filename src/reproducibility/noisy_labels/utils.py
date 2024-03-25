import lightning as L

# NOTE: THE FOLLOWING CODE SHOULD NOT BE USED IN ANY PROJECT
#       WE ARE USING IT ONLY TO ADHERE TO THE FLAWED METHODOLOGY
#       OF NOISY-LABELS BENCHMARK

class TestCallback(L.Callback):

    def __init__(self, test_freq: int = 1) -> None:
        super().__init__()
        self.freq = test_freq

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if (trainer.current_epoch+1) % self.freq != 0:
            return
        for batch in trainer.datamodule.test_dataloader()[0]:
            batch = [batch[0].to(trainer.model.device), batch[1].to(trainer.model.device)]
            trainer.model.test_step(batch, 0)