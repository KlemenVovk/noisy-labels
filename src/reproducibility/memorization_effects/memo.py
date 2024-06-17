# %%
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
# from torchvision.models.resnet import resnet34
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import ResNet34 as resnet34

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
noisy_path = "../../../data/noisylabels/CIFAR-10_human.pt"
noisy_labels = {k: torch.tensor(v) for k, v in torch.load(noisy_path).items()}
keys_to_test = ["aggre_label", "random_label1", "worse_label"]
noisy_labels

# %%
cifar10_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

clean_dataset = CIFAR10("../../../data/cifar", train=True, transform=cifar10_train_transform)
clean_labels = torch.tensor(clean_dataset.targets)
clean_labels

# %%
def generate_synthetic_labels(clean_labels, noisy_labels, seed=1337):
    # generate transition matrix T
    num_classes = 10
    T = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        clean_mask = clean_labels == i
        for j in range(num_classes):
            p = (noisy_labels[clean_mask] == j).float().mean()
            T[i, j] = p
    
    # sample the transition matrix
    gen = torch.Generator("cpu").manual_seed(seed)
    synthetic_labels = torch.multinomial(T[clean_labels], 1, generator=gen).squeeze()
    return synthetic_labels

generate_synthetic_labels(clean_labels, noisy_labels["aggre_label"])

# %%
def train_and_record_memorization(clean_labels, noisy_labels, num_epochs):
    dataset = clean_dataset
    dataset.targets = noisy_labels
    confidence_threshold = 0.95

    model = resnet34(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # use the same optimizer as in the CE config
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4)
    # use the same scheduler as in the CE config
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[60], 
        gamma=0.1)
    
    loader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=128, shuffle=False)
    
    clean_memos = []
    wrong_memos = []
    for epoch in range(num_epochs):
        probs = []
        for x, y in (pbar := tqdm(loader, ncols=80, desc=f"{epoch=:3}", leave=False)):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs.append(F.softmax(y_pred, -1))
            pbar.set_postfix({"loss": f"{loss.item():.2e}"})
        
        probs = torch.cat(probs)
        clean_idxs = clean_labels == noisy_labels        
        clean_frac = (probs[ clean_idxs].max(axis=-1)[0] > confidence_threshold).float().mean()
        wrong_frac = (probs[~clean_idxs].max(axis=-1)[0] > confidence_threshold).float().mean()
        clean_memos.append(clean_frac.item())
        wrong_memos.append(wrong_frac.item())
        scheduler.step()
    
    return clean_memos, wrong_memos

# %%
num_epochs = 150
memo_results = {}
for key in keys_to_test:
    print(f"\ntraining with {key}\n")
    memo_results[key] = {}
    
    human_labels = noisy_labels["aggre_label"]
    synth_labels = generate_synthetic_labels(clean_labels, human_labels)
    
    print(f"training human")
    human = train_and_record_memorization(clean_labels, human_labels, num_epochs)
    print(f"training synthetic")
    synth = train_and_record_memorization(clean_labels, synth_labels, num_epochs)

    memo_results[key]["human"] = {"clean": human[0], "wrong": human[1]}
    memo_results[key]["synthetic"] = {"clean": synth[0], "wrong": synth[1]}
torch.save(memo_results, "memo_results.pt")

# %%
memo_results = torch.load("memo_results.pt")

# %%
fig, ax = plt.subplots(2, 3, figsize=(10, 4))

for i, key in enumerate(keys_to_test):
    ax[0, i].set_title(f"clean labels ({key})")
    ax[0, i].plot(memo_results[key]["human"]["clean"], "b")
    ax[0, i].plot(memo_results[key]["synthetic"]["clean"], "b--")

    ax[1, i].set_title(f"wrong labels ({key})")
    ax[1, i].plot(memo_results[key]["human"]["wrong"], "r")
    ax[1, i].plot(memo_results[key]["synthetic"]["wrong"], "r--")

fig.supxlabel("epoch")
fig.supylabel("fraction of memorised samples")
plt.tight_layout()


