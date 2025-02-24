import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm

from utils import ResNet34 as resnet34

device = "cuda" if torch.cuda.is_available() else "cpu"

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "noise_label",
    type=str,
    help="noise label type",
    choices=["aggre_label", "random_label1", "worse_label"],
)
argparser.add_argument("--seed", type=int, default=1, help="random seed")
argparser.add_argument("--device", type=int, default=0, help="device")

args = argparser.parse_args()
seed = args.seed
noise_label = args.noise_label
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

noisy_path = "../../../data/noisylabels/CIFAR-10_human.pt"
noisy_labels = {k: torch.tensor(v) for k, v in torch.load(noisy_path).items()}


cifar10_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

clean_dataset = CIFAR10(
    "../../../data/cifar", train=True, transform=cifar10_train_transform
)
clean_labels = torch.tensor(clean_dataset.targets)
clean_labels


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


def train_and_record_memorization(clean_labels, noisy_labels, num_epochs):
    dataset = clean_dataset
    dataset.targets = noisy_labels
    confidence_threshold = 0.95

    model = resnet34(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # use the same optimizer as in the CE config
    optimizer = torch.optim.SGD(model.parameters(), lr=1, weight_decay=1e-4)
    # Xiu et al. 2021 use epoch//100 and train for 300 epochs
    # We need a smooth schedule and train for 150 epochs hence:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch / 50)
    )

    loader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
    )

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
        clean_frac = (
            (probs[clean_idxs].max(axis=-1)[0] > confidence_threshold).float().mean()
        )
        wrong_frac = (
            (probs[~clean_idxs].max(axis=-1)[0] > confidence_threshold).float().mean()
        )
        clean_memos.append(clean_frac.item())
        wrong_memos.append(wrong_frac.item())
        scheduler.step()

    return clean_memos, wrong_memos


num_epochs = 150
memo_results = {}
print(f"\ntraining with {noise_label} (seed {seed})\n")
memo_results = {}

human_labels = noisy_labels[noise_label]
synth_labels = generate_synthetic_labels(clean_labels, human_labels, seed=seed)

print("training human")
human = train_and_record_memorization(clean_labels, human_labels, num_epochs)
print("training synthetic")
synth = train_and_record_memorization(clean_labels, synth_labels, num_epochs)

memo_results["human"] = {"clean": human[0], "wrong": human[1]}
memo_results["synthetic"] = {"clean": synth[0], "wrong": synth[1]}
torch.save(memo_results, f"memo_results_{noise_label}_{seed}.pt")
