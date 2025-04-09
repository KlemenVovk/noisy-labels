import os
import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
import lightning as L

from noisypy.methods.classifiers.resnet import ResNet34 as resnet34


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "noise_label",
    type=str,
    help="noise label type",
    choices=["aggre_label", "random_label1", "worse_label"],
)
argparser.add_argument(
    "--scheduler", default="exponential", choices=["exponential", "step"]
)
argparser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
argparser.add_argument("--seed", type=int, default=1, help="random seed")
argparser.add_argument("--device", type=int, default=0, help="device")
argparser.add_argument(
    "--initial-lr", type=float, default=1, help="initial learning rate"
)

args = argparser.parse_args()
seed = args.seed
noise_label = args.noise_label

L.seed_everything(args.seed)

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.scheduler)
if args.weight_decay != 1e-4:
    save_dir += f"_wd_{args.weight_decay}"
if args.initial_lr != 1:
    save_dir += f"_lr_{args.initial_lr}"
os.makedirs(save_dir, exist_ok=True)

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

noisy_path = "../../../data/noisylabels/CIFAR-10_human.pt"
noisy_labels = {
    k: torch.tensor(v) for k, v in torch.load(noisy_path, weights_only=False).items()
}


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


def generate_pairflip_labels(clean_labels, noisy_labels, seed=1337):
    noise_rate = (clean_labels != human_labels).float().mean()
    # generate transition matrix T
    num_classes = 10
    T = torch.zeros(num_classes, num_classes)
    T[torch.eye(num_classes, dtype=torch.bool)] = 1 - noise_rate
    for i in range(num_classes):
        clean_mask = clean_labels == i
        # Estimate the original transition vector
        t = torch.zeros(num_classes)
        for j in range(num_classes):
            if i == j:
                continue
            p = (noisy_labels[clean_mask] == j).float().mean()
            t[j] = p
        # Select the most likely transition class
        j = t.argmax()
        T[i, j] = noise_rate

    assert torch.isclose(T.sum(axis=1), torch.ones(num_classes)).all()
    # sample the transition matrix
    gen = torch.Generator("cpu").manual_seed(seed)
    synthetic_labels = torch.multinomial(T[clean_labels], 1, generator=gen).squeeze()
    return synthetic_labels


def generate_symmetric_labels(clean_labels, noisy_labels, seed=1337):
    noise_rate = (clean_labels != noisy_labels).float().mean()
    # generate transition matrix T
    num_classes = 10
    T = torch.ones(num_classes, num_classes) * (noise_rate / (num_classes - 1))
    T[torch.eye(num_classes, dtype=torch.bool)] = 1 - noise_rate

    assert torch.isclose(T.sum(axis=1), torch.ones(num_classes)).all()
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
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay
    )
    if args.scheduler == "exponential":
        # Xiu et al. 2021 use epoch//100 and train for 300 epochs
        # We need a smooth schedule and train for 150 epochs hence:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch / 50)
        )
    elif args.scheduler == "step":
        # Use the original step scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50], gamma=0.1
        )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
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
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.2e}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        probs = torch.cat(probs)
        clean_mask = clean_labels == noisy_labels
        clean_frac = (
            (probs[clean_mask].max(axis=-1)[0] > confidence_threshold).float().mean()
        )
        wrong_frac = (
            (probs[~clean_mask].max(axis=-1)[0] > confidence_threshold).float().mean()
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

print("training human")
human = train_and_record_memorization(clean_labels, human_labels, num_epochs)
print("training synthetic (same T)")
synth_labels = generate_synthetic_labels(clean_labels, human_labels, seed=seed)
synth = train_and_record_memorization(clean_labels, synth_labels, num_epochs)
print("training pairflip")
pairflip_labels = generate_pairflip_labels(clean_labels, human_labels, seed=seed)
pairflip = train_and_record_memorization(clean_labels, pairflip_labels, num_epochs)
print("training symmetric")
symmetric_labels = generate_symmetric_labels(clean_labels, human_labels, seed=seed)
symmetric = train_and_record_memorization(clean_labels, symmetric_labels, num_epochs)

memo_results["human"] = {"clean": human[0], "wrong": human[1]}
memo_results["synthetic"] = {"clean": synth[0], "wrong": synth[1]}
memo_results["pairflip"] = {"clean": pairflip[0], "wrong": pairflip[1]}
memo_results["symmetric"] = {"clean": symmetric[0], "wrong": symmetric[1]}

# Save the results
torch.save(memo_results, f"{save_dir}/memo_results_{noise_label}_{seed}.pt")
