import torch
from torchvision import transforms
from torchvision.transforms.functional import erase


class Cutout(torch.nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, img):
        h, w = img.size(1), img.size(2)
        y = torch.randint(h, (1,))
        x = torch.randint(w, (1,))

        y1 = torch.clip(y - self.length // 2, 0, h).item()
        y2 = torch.clip(y + self.length // 2, 0, h).item()
        x1 = torch.clip(x - self.length // 2, 0, w).item()
        x2 = torch.clip(x + self.length // 2, 0, w).item()

        return erase(img, y1, x1, y2 - y1, x2 - x1, 0)


autoaug_paper_cifar10 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(
            policy=transforms.AutoAugmentPolicy.CIFAR10, fill=(0.4914, 0.4822, 0.4465)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(16),
    ]
)
