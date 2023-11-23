# Forward and backward loss correction training

Reimplementation of ["Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach"](https://arxiv.org/abs/1609.03683).

Basic idea:
1) Warmup phase: train a classifier, which is able to overfit the data, on data with noisy labels.
2) Estimate noise transition matrix: using the trained classifier, predict labels for all the training examples and estimate the noise transition matrix $T$.
3) Retrain: retrain the model with noise corrected loss.

The starter code was taken from the [original tensorflow implementation](https://github.com/giorgiop/loss-correction) and converted and optimised into pytorch([link to file](../utils/forward_backward_T.py)).

To confirm that losses match the baseline, we check whether the loss output matches for multiple randomly generated inputs.
This approach might leave some room for error, since we didn't match the loss/acc curves of both implementations (converting and running unfamiliar legacy tensorflow code seemed like too much of a hassle:3), but the behaviour seems to match the expectations - marginally higher than the baseline performance, with expected loss jumps after warmup.
