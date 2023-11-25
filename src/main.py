import lightning as L
from aim.pytorch_lightning import AimLogger
import yaml
import argparse
import os

########################################################################
# TODO: automatically build from config - utils/config_builder.py
# TODO: figure out a cleaner way than just importing * from 
#       from modules as this will not automatically work for custom
#       modules added in the future
# TODO: try this: perhaps, this idea of yaml configs is not as good
#       - if you have some sort of object defined config, you could
#       define a basic config, and then extend it and only
#       change the necessary stuff for new experiment/run
########################################################################

# importing * only to populate registries
from data.datamodules.basic import *
from data.datamodules.noise import *
from data.datasets.cifar10 import *
from data.noise.noises import *
from data.transforms.cifar10 import *
from models.classifiers.resnet import *
from models.learning_strategies.cores2.cores2 import *
from utils.registry import REGISTER

def main(args):
    L.seed_everything(args["seed"], workers=True)

    strats = REGISTER["strategies"]
    classifiers = REGISTER["classifiers"]
    datasets = REGISTER["datasets"]
    noises = REGISTER["noises"]
    datamodules = REGISTER["datamodules"]
    transforms = REGISTER["transforms"]

    ####################
    # get module classes
    ####################

    # classifier - e.g. resnet34
    classifier_cls = classifiers[args["classifier"]]
    classifier_kws = args["classifier_args"]

    # learning strategy - e.g. cores2
    strat_cls = strats[args["strategy"]]
    strat_kws = args["strategy_args"]

    # noise
    try: # this happens when you build configs by hand heh
        noise_cls = noises[args["datamodule_args"].pop("noise")]
        noise_args = args["datamodule_args"].pop("noise_args")
        noise = noise_cls(**noise_args)
    except KeyError:
        noise = None

    # datamodule
    datamodule_cls = datamodules[args["datamodule"]]
    datamodule_kws = args["datamodule_args"]

    # transforms
    train_transform = transforms[args["train_dataset_args"].pop("transform")]
    val_transform = transforms[args["val_dataset_args"].pop("transform")]

    # datasets
    train_dataset_cls = val_dataset_cls = test_datset_cls = datasets[args["dataset"]]
    dataset_args = args["dataset_args"]
    train_kws = {"transform": train_transform, **dataset_args, **args["train_dataset_args"]}
    val_kws = {"transform": val_transform, **dataset_args, **args["val_dataset_args"]}
    test_kws = val_kws # TODO extend

    ##############
    # init modules
    ##############
    datamodule_kws["train_dataset_cls"] = train_dataset_cls
    datamodule_kws["train_dataset_kws"] = train_kws
    datamodule_kws["val_dataset_cls"] = val_dataset_cls
    datamodule_kws["val_dataset_kws"] = val_kws
    datamodule_kws["test_dataset_cls"] = test_datset_cls
    datamodule_kws["test_dataset_kws"] = test_kws
    if noise is not None: datamodule_kws["noise"] = noise

    datamodule = datamodule_cls(**datamodule_kws)

    strategy = strat_cls(
        classifier_cls, classifier_kws,
        datamodule,
        **strat_kws)
    
    ##############
    # init trainer
    ##############

    trainer_args = args["trainer_args"]

    aim_logger = AimLogger(
        experiment=args["strategy"],
        train_metric_prefix='train_',
        test_metric_prefix='test_',
        val_metric_prefix='val_',
    )

    trainer = L.Trainer(logger=aim_logger, deterministic=True, **trainer_args) # logger is fixed as it should be always the same
    trainer.fit(strategy, datamodule)


def parse_config(config_path):
    config = None
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file {config_path} not found.')
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Sanity checks
    #assert "model_name" in config, "Config file must contain the model_name."
    #assert "dataset_name" in config, "Config file must contain the dataset_name."
    #assert "seed" in config, "Config file must contain the seed."
    #assert isinstance(config["seed"], int), "Seed must be an int."
    #assert "model_args" in config, "Config file must contain the model_args section."
    #assert "dataset_args" in config, "Config file must contain the dataset_args section."
    #assert "trainer_args" in config, "Config file must contain the trainer_args section."
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the hyperparameters yaml file.', default="configs/cores_cifar10_noise.yaml", required=False)
    #parser.add_argument('config', type=str, help='Path to the hyperparameters yaml file.')
    args = parser.parse_args()
    hyperparameters = parse_config(args.config)
    hyperparameters["hp_file"] = args.config # save the path to the hyperparameters file so that we can quickly reference it from the aim logger
    main(hyperparameters)

