import types

from numpy import ndarray
import PIL.Image as Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS


def reset_resnet_layer_parameters(resnet_layer: nn.Module):
    for block in resnet_layer.children():
        for layer in block.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    # set requires_grad to True
    for param in resnet_layer.parameters():
        param.requires_grad = True


def renew_layers(
    model: nn.Module,
    last_num_layers: int,
    model_class: str = "pytorch_resnet",
    num_classes=10,
):
    if model_class == "pytorch_resnet":
        if last_num_layers >= 3:
            print("re-initalize block 2")
            reset_resnet_layer_parameters(model.layer2)

        if last_num_layers >= 2:
            print("re-initalize block 3")
            reset_resnet_layer_parameters(model.layer3)

        if last_num_layers >= 1:
            print("re-initalize block 4")
            reset_resnet_layer_parameters(model.layer4)

        print("re-initalize the final layer")
        model.fc = nn.Linear(512, num_classes)
    elif model_class == "paper_resnet":
        if last_num_layers >= 3:
            print("re-initalize block 2")
            reset_resnet_layer_parameters(model.layer2)

        if last_num_layers >= 2:
            print("re-initalize block 3")
            reset_resnet_layer_parameters(model.layer3)

        if last_num_layers >= 1:
            print("re-initalize block 4")
            reset_resnet_layer_parameters(model.layer4)

        print("re-initalize the final layer")
        model.linear = nn.Linear(512, num_classes)
    else:
        raise NotImplementedError(f"model class {model_class} is not implemented yet.")

    return model


def update_dataloader(datamodule: LightningDataModule, dataset: Dataset):
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return [
            DataLoader(
                dataset,
                self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )
        ]

    datamodule.train_dataloader = types.MethodType(train_dataloader, datamodule)


def update_train_data_and_criterion(
    model: nn.Module,
    train_data: ndarray,
    noisy_targets: torch.LongTensor,
    transform_train: transforms.Compose,
    batch_size: int = 128,
    device: torch.device = torch.device("cuda"),
):
    predict_dataset = Semi_Unlabeled_Dataset(train_data, transform_train)
    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    soft_outs = predict_softmax(predict_loader, model, device=device)
    confident_indexs = split_confident(soft_outs, noisy_targets)
    confident_dataset = Train_Dataset(
        train_data[confident_indexs], noisy_targets[confident_indexs], transform_train
    )

    # compute class weights
    bin_counts = torch.bincount(noisy_targets[confident_indexs], minlength=10)
    class_weights = torch.zeros_like(bin_counts, dtype=torch.float32)
    class_weights[bin_counts != 0] = (
        bin_counts[bin_counts != 0].float().mean() / bin_counts[bin_counts != 0]
    )
    # initialize weighted cross entropy loss
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    return confident_dataset, criterion


def predict_softmax(
    predict_loader: DataLoader,
    model: nn.Module,
    device: torch.device = torch.device("cuda"),
):
    model.eval()
    softmax_outs = []
    with torch.no_grad():
        for images1, images2 in tqdm(predict_loader, desc="Predicting", leave=False):
            logits1 = model(images1.to(device))
            logits2 = model(images2.to(device))
            outputs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
            softmax_outs.append(outputs)

    return torch.cat(softmax_outs, dim=0).cpu()


def split_confident(outs: torch.Tensor, noisy_targets: torch.Tensor):
    _, preds = torch.max(outs.data, 1)
    confident_idx = torch.where(noisy_targets == preds)[0]
    return confident_idx.tolist()


# Custom Datasets
class Train_Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.train_data = data
        self.train_labels = labels
        self.length = len(self.train_labels)
        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data, self.train_labels


class Semi_Unlabeled_Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.train_data = data
        self.length = self.train_data.shape[0]

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img = self.train_data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            out1 = self.transform(img)
            out2 = self.transform(img)

        return out1, out2

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data
