
## Environment

Using conda:
```bash
conda create -f environment.yml
```

In case something fails, here are all the commands needed:
```bash
conda create -n noisylabels python=3.10
conda activate noisylabels
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge lightning aim
aim init
```

## How to run?
The model and dataset to use along with all the hyperparameters is defined in a single yaml file inside the `src/runs` folder. The following structure is enforced:
- `model_name: str` - needed to define what model to use (see models dictionary in main.py for valid keys)
- `dataset_name: str` - needed to define what dataset to use (see datamodules dictionary in main.py for valid keys)
- `dataset_args: dict` - all hyperparameters passed to the dataset (LightninigDataModule from the `src/data` folder)
- `model_args: dict` - all hyperparameters passed to the model class (LightningModule from the `models` folder)
- `trainer_args: dict` - all hyperparameters passed to the Lightning trainer (see trainer in main.py)
- `seed: int` - enforced for reproducibility
An example yaml file can be found [here](https://github.com/KlemenVovk/noisy-labels/blob/master/src/runs/cores2_cifar10.yaml).

Once you have defined the file (or decided which one to use from the premade ones in `src/runs`), run the script with
To run the training and evaluation (make sure that you are running from the conda environment):
```bash
python main.py path/to/hyperparameters.yaml
```

For logging we are using a lightweight, but very powerful local logger called [Aim](https://aimstack.readthedocs.io/en/latest/) which integrates natively with Lightning. To run the logger's web UI (make sure that you are running from the conda environment):
```bash
aim init # only needed the first time around
aim up
```
The UI should be accessible at [http://127.0.0.1:43800/](http://127.0.0.1:43800/).

## Repository structure

The repository is more or less following the [Data Science Cookie Cutter project structure](https://drivendata.github.io/cookiecutter-data-science/)

Briefly:
- `.aim` - logger storage
- `data` - stores the raw datasets (cifar automatically downloaded on run)
- `docs` - documentation about specific methods (mostly non-trivial implementation choices and differences between code and papers)
- `src` - everything code
  - `data` - Lightning data modules for working with data.
  - `models` - models and everything regarding their training
  - `runs` - contains yaml files with hyperparameters for running experiments
  - `utils` - currently contains implementations of losses and noise generation directly from the authors in order to see if we can reproduce at all, but basically everything in here should be rewritten and tested to be the same. The goal is to get rid of this (replace with our own implementation that fits in the framework nicely). The reason we're keeping this separate is so that foreign code we intend to delete is all in one place and not intertwined with out code.
  - `main.py` - a short script that runs the actual training
