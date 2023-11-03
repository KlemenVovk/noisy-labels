import lightning as L
from models.cores2 import SampleSieve
from data.cifar import CIFAR10DataModule
from utils.cores2 import train_cifar10_transform, test_cifar10_transform
from aim.pytorch_lightning import AimLogger
import yaml
import argparse
import os

# TODO: test instead of val
# To add a new model, add the relevant classes/transforms to the following dictionaries. That's it!
models = {
    'cores2': SampleSieve,
}

datamodules = {
    'cifar10': CIFAR10DataModule,
}

# TODO: What will we do with transforms? Is something like this ok?
# My proposal is to automatically determine the correct transforms.
# So if someone runs the cores2 model with cifar10 dataset, the train transform is then simply train_transforms['cores2_cifar10'].
train_transforms = {
    'cores2_cifar10': train_cifar10_transform,
}
test_transforms = {
    'cores2_cifar10': test_cifar10_transform,
}

def main(args):
    L.seed_everything(args["seed"])
    model_cls = models[args["model_name"]]
    data_module_cls = datamodules[args["dataset_name"]]

    model_args = args["model_args"]
    trainer_args = args["trainer_args"]
    dataset_args = args["dataset_args"]
    dataset_args["train_transform"] = train_transforms[args["model_name"] + "_" + args["dataset_name"]]
    dataset_args["test_transform"] = test_transforms[args["model_name"] + "_" + args["dataset_name"]]


    aim_logger = AimLogger(
        experiment=args["model_name"],
        train_metric_prefix='train_',
        test_metric_prefix='test_',
        val_metric_prefix='val_',
    )

    data_module = data_module_cls(**dataset_args)
    data_module.setup()
    model = model_cls(**model_args, datamodule=data_module)
    trainer = L.Trainer(logger=aim_logger, **trainer_args)
    trainer.fit(model, data_module)


def parse_hyperparameters(hyperparameters_path):
    hyperparameters = None
    if not os.path.exists(hyperparameters_path):
        raise FileNotFoundError(f'Hyperparameters file {hyperparameters_path} not found.')
    
    with open(hyperparameters_path, 'r') as f:
        hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
    
    # Sanity checks
    assert "model_name" in hyperparameters, "Hyperparameters file must contain the model_name."
    assert "dataset_name" in hyperparameters, "Hyperparameters file must contain the dataset_name."
    assert "seed" in hyperparameters, "Hyperparameters file must contain the seed."
    assert isinstance(hyperparameters["seed"], int), "Seed must be an int."
    assert "model_args" in hyperparameters, "Hyperparameters file must contain the model_args section."
    assert "dataset_args" in hyperparameters, "Hyperparameters file must contain the dataset_args section."
    assert "trainer_args" in hyperparameters, "Hyperparameters file must contain the trainer_args section."
    return hyperparameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hyperparameters', type=str, help='Path to the hyperparameters yaml file.')
    args = parser.parse_args()
    hyperparameters = parse_hyperparameters(args.hyperparameters)
    hyperparameters["hp_file"] = args.hyperparameters # save the path to the hyperparameters file so that we can quickly reference it from the aim logger
    main(hyperparameters)

