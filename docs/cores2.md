- The authors use a different resnet. Using Pytorch's implementation of resnet, we achieve ~10% less accuracy across the board.
- The paper states `We first train network on the dataset for 10 warm-up epochs with only CE (Cross Entropy) loss. Then β is linearly increased from 0 to 2 for next 30 epochs and kept as 2 for the rest of the epochs.` however, looking at the loss function they provided, only CE is used for 30 epochs and then beta starts off at 2.
- The paper states `initial learning rate (0.1)` for CIFAR 10, however, looking at the code, the initial learning rate is 0.05.
- The learning rate is adjusted as
```python
def f_beta(epoch):
    beta1 = np.linspace(0.0, 0.0, num=10)
    beta2 = np.linspace(0.0, 2, num=30)
    beta3 = np.linspace(2, 2, num=60)
 
    beta = np.concatenate((beta1,beta2,beta3),axis=0) 
    return beta[epoch]

alpha_plan = [0.1] * 50 + [0.01] * 50 
lr=alpha_plan[epoch]/(1+f_beta(epoch)) # each epoch
```
- The paper states `momentum (0.9), weight decay (0.0005)`, however, the code adjusts only the learning rate `optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)` of the SGD optimizer keeping others default, so 0 momentum and 0 weight decay.