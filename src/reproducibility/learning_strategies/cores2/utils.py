import numpy as np
import torch
import torch.nn.functional as F

# tested against original, produces same results (up to RNG, since torch rngs are used instead of numpy)
def noisify_instance(x, y, noise_rate):
    x = x.flatten(start_dim=1) # [B, C*H*W]
    n_samples, feature_dim = x.shape
    num_classes = y.max() + 1

    # Probability that instance's label is flipped (to any label)
    instance_flip_rates = torch.nn.init.trunc_normal_(torch.empty(n_samples), mean=noise_rate, std=0.1, a=0, b=1) # [B]

    # Probability that instance's label is flipped to each label (not really a probability)
    instance_label_noise = torch.normal(0, 1, size=(feature_dim, num_classes)) # [C*H*W, CLS]

    p = x @ instance_label_noise # [B, FEAT] @ [FEAT, CLS] -> [B, CLS]
    p[np.arange(p.shape[0]), y] = -torch.inf # mask softmax
    p = instance_flip_rates.unsqueeze(-1) * F.softmax(p, dim=-1) # [B, 1] * [B, CLS] -> [B, CLS]
    p[np.arange(p.shape[0]), y] = 1 - instance_flip_rates # this is actually wrong since p[true_class] will not be 1 - instance_flip_rates after normalization
    p /= p.sum(axis=1, keepdim=True) # normalise rows
    y_noise = torch.multinomial(p, 1).flatten() # batch sample multinomial distributions (rows of p are individual multinomials)
    return y_noise, torch.mean((y != y_noise).float())


# NOTE: this is from the authors just to test, below is our implementation
def noisify_instance_original(train_data,train_labels,noise_rate):
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
    instance_flip_rates = torch.nn.init.trunc_normal_(torch.empty(n_samples), mean=noise_rate, std=0.1, a=0, b=1)
    np.random.normal(noise_rate, )
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
