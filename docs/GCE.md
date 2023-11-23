# Generalized crossentropy / Truncated loss

Reimplementation of ["Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"](https://arxiv.org/abs/1805.07836).

Basic idea: During training assign weights (0 or 1) to indiviudal training examples. For CIFAR, this means 50.000 weights. Based on therse the loss is calculated. If weight is 1, the loss for this example gets calculated from the predicted logits of this example, while if it is 0, it gets set to a constant based on the hyperparameters. At first these weights are all 1s. After some warmup epochs, start the "pruning"(update the example weights) process every k steps. At pruning step, the training examples, for which the model's loss is less than a treshold, get assigned weight 0. Note, that the model through which we get these losses is the model with highest validation accuracy up to that epoch.

The starter code was taken from the [unofficial implementation](https://github.com/AlanChou/Truncated-Loss) and was optimised (removed numpy calls, used implicit broadcasting)([link to file](../utils/GCE.py)).

To confirm that losses match the baseline, we check whether the loss output matches for multiple randomly generated inputs.
Here we also match the train/test loss and acc curves with the original implementation.