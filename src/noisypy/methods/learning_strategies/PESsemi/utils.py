import types

from numpy import ndarray
from tqdm import tqdm
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS


def linear_rampup(current: float, warm_up: int =20, rampup_length: int=16, lambda_u: float=5):
    current = torch.clip(torch.tensor((current - warm_up) / rampup_length), 0.0, 1.0)
    return lambda_u * current.item()


def update_dataloaders(datamodule: LightningDataModule, labeled_dataloader: DataLoader, unlabeled_dataloader: DataLoader):
    # This function is used to change the dataloaders of the datamodule.
    # While it might not be the prettiest solution, it is simple and works.
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return [labeled_dataloader, unlabeled_dataloader]
    datamodule.train_dataloader = types.MethodType(train_dataloader, datamodule)


def update_train_data_and_criterion(model: Module, train_data: ndarray, noisy_targets: torch.LongTensor, 
                                    transform_train: transforms.Compose, batch_size: int=128, device: torch.device=torch.device('cuda')):
    predict_dataset = Semi_Unlabeled_Dataset(train_data, transform_train)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=batch_size * 2, 
                                shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model, device=device)

    confident_indexs, unconfident_indexs = split_confident(soft_outs, noisy_targets)
    confident_dataset = Semi_Labeled_Dataset(train_data[confident_indexs], noisy_targets[confident_indexs], transform_train)
    unconfident_dataset = Semi_Unlabeled_Dataset(train_data[unconfident_indexs], transform_train)

    uncon_batch = int(batch_size / 2) if len(unconfident_indexs) > len(confident_indexs) else int(len(unconfident_indexs) / (len(confident_indexs) + len(unconfident_indexs)) * batch_size)
    con_batch = batch_size - uncon_batch

    labeled_trainloader = DataLoader(dataset=confident_dataset, batch_size=con_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    unlabeled_trainloader = DataLoader(dataset=unconfident_dataset, batch_size=uncon_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # compute class weights
    bin_counts = torch.bincount(noisy_targets[confident_indexs], minlength=10)
    class_weights = torch.zeros_like(bin_counts, dtype=torch.float32)
    class_weights[bin_counts != 0] = bin_counts[bin_counts != 0].float().mean() / bin_counts[bin_counts != 0]
    class_weights[class_weights > 3] = 3.
    return labeled_trainloader, unlabeled_trainloader, class_weights.to(device)


def predict_softmax(predict_loader: DataLoader, model: Module, device: torch.device = torch.device('cuda')):
    model.eval()
    softmax_outs = []
    with torch.no_grad():
        for images1, images2 in tqdm(predict_loader, desc=f'Predicting', leave=False):
            logits1 = model(images1.to(device)) 
            logits2 = model(images2.to(device))
            outputs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
            softmax_outs.append(outputs)

    return torch.cat(softmax_outs, dim=0).cpu()


def split_confident(outs: torch.Tensor, noisy_targets: torch.Tensor):
    _, preds = torch.max(outs.data, 1)
    confident_idx = torch.where(noisy_targets == preds)[0]
    unconfident_idx = torch.where(noisy_targets != preds)[0]
    return confident_idx.tolist(), unconfident_idx.tolist()


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
    

class Semi_Labeled_Dataset(Dataset):
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
            out1 = self.transform(img)
            out2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return out1, out2, target

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
