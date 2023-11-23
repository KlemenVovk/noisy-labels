# Baseline Model trained with cross entropy loss

This is the only method included in the official starter code for noisylabels, so we can confirm that our reimplementation produces same results (same train/val loss and acc curves).

The main problem is that the baseline performance is way higher than that reported in the paper.
We don't know the reason for this. Since methods in the paper lack any clarification about selecting the scores, we use same procedure for getting the final test score as they mentioned in this [github issue](https://github.com/UCSC-REAL/cifar-10-100n/issues/5#issuecomment-1471190937).

Another possible reason for higher performance is the use of "custom" implementation of resnet34, which differs from the original paper and pytorch implementation, as it does not reduce the feature maps' dimensions as much, meaning it propagates wider features, which enable better predictive performance.
