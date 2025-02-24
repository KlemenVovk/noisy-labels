from argparse import ArgumentParser

from configs.benchmark_configs.cifar100n import method_configs
from utils import update_config, TestCallback
from noisypy.configs.data.cifar100n import (
    cifar100n_clean_benchmark_config,
    cifar100n_noisy_benchmark_config,
)


data_configs = {
    "clean": cifar100n_clean_benchmark_config,
    "noisy": cifar100n_noisy_benchmark_config,
}


if __name__ == "__main__":
    parser = ArgumentParser("Noisy labels runner.")
    parser.add_argument("method_config", choices=method_configs.keys())
    parser.add_argument("data_config", choices=data_configs.keys())
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    print(
        f"Running {args.method_config} on {args.data_config} with seed {args.seed}..."
    )

    config = update_config(
        method_configs[args.method_config],
        data_configs[args.data_config],
        args.seed,
        logdir="../logs/benchmark",
    )
    model, datamodule, trainer = config.build_modules()

    print(f"Train samples: {datamodule.num_train_samples}")
    print(f"Val   samples: {datamodule.num_val_samples}")
    print(f"Test  samples: {datamodule.num_test_samples}")
    trainer.callbacks.append(TestCallback(test_freq=1))
    trainer.fit(model, datamodule)
