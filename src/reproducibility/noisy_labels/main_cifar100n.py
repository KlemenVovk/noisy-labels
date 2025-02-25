from argparse import ArgumentParser
import torch
from noisypy.configs.data.cifar100n import (
    cifar100n_clean_config,
    cifar100n_noisy_config,
)

from configs.cifar100n import *
from utils import update_config, TestCallback


method_configs = {
    "CAL": CAL_config,
    "CE": CE_config,
    "co_teaching": co_teaching_config,
    "co_teaching_plus": co_teaching_plus_config,
    "divide_mix": divide_mix_config,
    "divide_mix_clean": divide_mix_clean_config,
    "ELR": ELR_config,
    "ELR_plus": ELR_plus_config,
    "PES_semi": PES_semi_config,
    "SOP": SOP_config,
    "SOP_plus": SOP_plus_config,
    "volminnet": volminnet_config,
}

data_configs = {"clean": cifar100n_clean_config, "noisy": cifar100n_noisy_config}


if __name__ == "__main__":
    parser = ArgumentParser("Noisy labels runner.")
    parser.add_argument("method_config", choices=method_configs.keys())
    parser.add_argument("data_config", choices=data_configs.keys())
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    print(
        f"Running {args.method_config} on {args.data_config} with seed {args.seed}..."
    )

    config = update_config(
        method_configs[args.method_config], data_configs[args.data_config], args.seed
    )
    model, datamodule, trainer = config.build_modules()

    if args.synthetic:
        clean_labels = torch.tensor(datamodule.train_datasets[0].targets)
        noisy_labels = torch.tensor(datamodule.train_datasets[0].noisy_targets)
        print(noisy_labels)

        num_classes = 100
        T = torch.zeros(num_classes, num_classes)

        for i in range(num_classes):
            clean_mask = clean_labels == i
            for j in range(num_classes):
                p = (noisy_labels[clean_mask] == j).float().mean()
                T[i, j] = p
        print(T)
        seed = args.seed if args.seed is not None else 42
        torch.manual_seed(seed)
        synthetic_labels = torch.multinomial(T[clean_labels], 1).squeeze()

        for train_dataset in datamodule.train_datasets:
            train_dataset.noisy_targets = synthetic_labels

        print(datamodule.train_datasets[0].noisy_targets)
        print(
            f"noise rate: {(clean_labels != datamodule.train_datasets[0].noisy_targets).float().mean()}"
        )

        trainer.loggers[0]._root_dir = trainer.loggers[0]._root_dir.replace(
            "logs", "logs/synthetic"
        )
        trainer.loggers[0]._save_dir = trainer.loggers[0]._save_dir.replace(
            "logs", "logs/synthetic"
        )

    print(f"Train samples: {datamodule.num_train_samples}")
    print(f"Val   samples: {datamodule.num_val_samples}")
    print(f"Test  samples: {datamodule.num_test_samples}")
    trainer.callbacks.append(TestCallback(test_freq=1))
    trainer.fit(model, datamodule)
