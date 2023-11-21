# Generalized crossentropy / Truncated loss

Reimplementation of ["Does Label Smoothing Mitigate Label Noise?"](http://proceedings.mlr.press/v119/lukasik20a/lukasik20a.pdf)([supplementary material](http://proceedings.mlr.press/v119/lukasik20a/lukasik20a-supp.pdf)).

There is no starter code here, as the whole method amounts to only adding label smoothing to the crossentropy baseline.

As there are no known reimplementations of this specific paper, we rely only on results, which seem to be marginally better than the baseline crossentropy training with no label smoothing.
We also lack the knowledge of the smoothing hyperparameter used in the noisylabels experiments.
Therefore we ran multiple experiments, with smoothing amount slightly larger, than the noise amount present in the dataset, where each of the runs seemed to perform around vanilla CE (around 0.5% better on CIFAR10-N random label1, with the non-pytorch resnet).
The results are closer to the vanilla CE than shown in the paper, so further investigation might be needed.