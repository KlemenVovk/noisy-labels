import lightning as L
from lightning.pytorch.loggers import CSVLogger


def update_config(base_config, new_data_config, new_seed):
    class UpdatedConfig(base_config):
        data_config = new_data_config if base_config._data_config_wrapper is None else base_config._data_config_wrapper(new_data_config)
        seed = new_seed
        trainer_args = {
            **base_config.trainer_args,
            "logger": CSVLogger(f"../logs/{new_data_config.__name__}", name=f"{base_config.__name__}_{new_seed}"),
        }
    return UpdatedConfig


class ConfigIter:

    def __init__(self, config, data_configs, seeds):
        self.config = config
        self.data_configs = data_configs
        self.seeds = seeds
    
    def __iter__(self):
        for data_config in self.data_configs:
            for seed in self.seeds:
                yield update_config(self.config, data_config, seed) 


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