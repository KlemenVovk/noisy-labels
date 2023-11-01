import numpy as np
import torch
import torch.nn.functional as F

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
        # Make the probability of flipping to the correct class 1 - instance_flip_rate
        p[y_n] = 1 - instance_flip_rate
        # Normalize to get probabilities
        p /= p.sum()
        # Randomly choose a label according to the probabilities p
        noisy_labels.append(np.random.choice(num_classes, p=p))
    return torch.tensor(noisy_labels)

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    n_classes = 10
    n_samples = 15
    x = np.random.normal(0,1,size=(n_samples, 100, 100, 3))
    y = np.random.choice(n_classes,size=n_samples)
    print(generate_instance_dependent_noise(x,y,0.2, n_classes))
