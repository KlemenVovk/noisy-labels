
## Environment

Using conda:
```bash
conda create -n noisylables python=3.10
conda activate noisylabels
pip install .
```

## How to run?

TODO


## Repository structure

The repository is more or less following the [Data Science Cookie Cutter project structure](https://drivendata.github.io/cookiecutter-data-science/)

Briefly:
- `src` - Everything code
  - `noisypy` - Python package for working with LNL methods.
  - `reproducibility` - Experiments for reproducibility challenge submission.
    - `learning_strategies` - Experiments to verify our implementations for each LNL method.
    - `logs` - Results of our experiments.
    - `memorization_effects` - Memorization effects experiment - see `memorization.ipynb` and `memo.py`.
    - `noise_hypothesis_testing` - Noise clustering hypothesis testing experiment - see `hypothesis_testing.ipynb`.
    - `noisy_labels` - Benchmark reproduction. `main.py` is for normal CIFAR-10N human and synthetic runs (`--synthetic` switch to run synthetic versions). `main_cifar100n.py` same thing for CIFAR-100N. `main_paper.py` runs CIFAR-10N authors claimed configs (with fixed learning rate, schedulers and optimizers).
- `tests` - Unit tests for some of the package modules.
