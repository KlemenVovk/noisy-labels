# Run from the src dir: python reproducibility/cores2.py
import sys; sys.path.append(".")
from main import parse_hyperparameters, main

# TODO: synthetic noise
if __name__ == "__main__":
    run_hparams_paths = ["runs/cores2_cifar10_instance_02.yaml", "runs/cores2_cifar10_instance_04.yaml", "runs/cores2_cifar10_instance_06.yaml"]
    for run_hparams_path in run_hparams_paths:
        hyperparameters = parse_hyperparameters(run_hparams_path)
        hyperparameters["hp_file"] = run_hparams_path # save the path to the hyperparameters file so that we can quickly reference it from the aim logger
        main(hyperparameters)