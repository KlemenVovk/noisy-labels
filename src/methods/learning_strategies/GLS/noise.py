import torch

chosen_noise_file = "/home/klemen/projects/negative-label-smoothing/traindata_0.2.pt"
chosen_noise = torch.load(chosen_noise_file)

def lambda_gls_noise(feature, target, index):
    return int(chosen_noise["noisy_labels"][index])

