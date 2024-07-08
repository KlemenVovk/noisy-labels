# Reproduction of noisy-labels benchmark entries in our framework.

We try to match loss and accuracy curves of the original implementations with their default parameters.

Structure: each method has its own folder. Inside folder, there are:

* utils.py - original models etc.
* assets folder - dumps from original runs - e.g. loss and accuracy curves, dumped noisy labels etc.
* config files that reproduce the original runs 