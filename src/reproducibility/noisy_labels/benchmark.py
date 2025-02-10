from argparse import ArgumentParser
import torch

from configs.benchmark_configs.cifar10n import *
from utils import update_config, TestCallback
from noisypy.configs.data.cifar10n import (
    cifar10n_clean_benchmark_config,
    cifar10n_aggre_benchmark_config,
    cifar10n_worse_benchmark_config,
    cifar10n_random1_benchmark_config,
    cifar10n_random2_benchmark_config,
    cifar10n_random3_benchmark_config,
)

method_configs = {
    "CAL": CAL_config,
    "CE": CE_config,
    "co_teaching": co_teaching_config,
    "co_teaching_plus": co_teaching_plus_config,
    "cores2": cores2_config,
    "divide_mix": divide_mix_config,
    "divide_mix_clean": divide_mix_clean_config,
    "ELR": ELR_config,
    "ELR_plus": ELR_plus_config,
    "GCE": GCE_config,
    "jocor": jocor_config,
    "PES": PES_config,
    "PES_semi": PES_semi_config,
    "SOP": SOP_config,
    "SOP_plus": SOP_plus_config,
    "volminnet": volminnet_config,
}

data_configs = {
    "clean": cifar10n_clean_benchmark_config,
    "aggre": cifar10n_aggre_benchmark_config,
    "worse": cifar10n_worse_benchmark_config,
    "random1": cifar10n_random1_benchmark_config,
    "random2": cifar10n_random2_benchmark_config,
    "random3": cifar10n_random3_benchmark_config,
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
        method_configs[args.method_config], data_configs[args.data_config], args.seed
    )
    model, datamodule, trainer = config.build_modules()

    print(f"Train samples: {datamodule.num_train_samples}")
    print(f"Val   samples: {datamodule.num_val_samples}")
    print(f"Test  samples: {datamodule.num_test_samples}")
    trainer.callbacks.append(TestCallback(test_freq=1))
    trainer.fit(model, datamodule)
