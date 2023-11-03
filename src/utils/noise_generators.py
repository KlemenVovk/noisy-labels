import numpy as np
import torch
import torch.nn.functional as F

# TODO DELETE: this is from the authors just to test, below is our implementation
def noisify_instance(train_data,train_labels,noise_rate):
    if max(train_labels)>10:
        num_class = 100
    else:
        num_class = 10
    np.random.seed(0)

    q_ = np.random.normal(loc=noise_rate,scale=0.1,size=1000000)
    q = []
    for pro in q_:
        if 0 < pro < 1:
            q.append(pro)
        if len(q)==50000:
            break

    w = np.random.normal(loc=0,scale=1,size=(32*32*3,num_class))

    noisy_labels = []
    for i, sample in enumerate(train_data):
        sample = sample.flatten()
        p_all = np.matmul(sample,w)
        p_all[train_labels[i]] = -1000000
        p_all = q[i]* F.softmax(torch.tensor(p_all),dim=0).numpy()
        p_all[train_labels[i]] = 1 - q[i]
        noisy_labels.append(np.random.choice(np.arange(num_class),p=p_all/sum(p_all)))
    over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum())/50000
    return noisy_labels, over_all_noise_rate

# This is our implementation of the above algorithm, needs testing, so currently we are using the authors' implementation
def generate_instance_dependent_noise(x: torch.Tensor,y: torch.Tensor, noise_rate:float, num_classes: int) -> torch.Tensor:
    feature_dim = x[0].flatten().shape[0]
    n_samples = len(y)

    # Probability that instance's label is flipped (to any label)
    instance_flip_rates = np.random.normal(noise_rate, 0.1, size=n_samples).clip(0, 1)
    # Probability that instance's label is flipped to each label (not really a probability)
    instance_label_noise = np.random.normal(0, 1, size=(feature_dim, num_classes))
    noisy_labels = []
    for n in range(n_samples):
        x_n, y_n = x[n].flatten(), y[n]
        instance_flip_rate = instance_flip_rates[n]
        # Probability of flipping to each label
        p = x_n.dot(instance_label_noise)
        # Don't consider flipping to the correct class
        p[y_n] = -np.inf
        p = instance_flip_rate * F.softmax(torch.tensor(p), dim=0)
        # Make the "probability" of keeping the correct class 1 - instance_flip_rate
        p[y_n] = 1 - instance_flip_rate
        # Normalize to get actual probabilities
        p /= p.sum()
        # Randomly choose a label according to the probabilities p
        noisy_labels.append(np.random.choice(num_classes, p=p))
    return torch.tensor(noisy_labels)
