import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
from tqdm import tqdm

from ..base import LearningStrategyModule
from .utils import GCELoss, Mixup


class DISC(LearningStrategyModule):
    def __init__(
        self,
        datamodule,
        classifier_cls,
        classifier_args,
        optimizer_cls,
        optimizer_args,
        scheduler_cls,
        scheduler_args,
        start_epoch: int,
        alpha: float,  #
        sigma: float,  # 0.5
        momentum: float,  # 0.95
        lambd_ce: float,  # 1
        lambd_h: float,  # 1
        *args,
        **kwargs,
    ):
        super().__init__(
            datamodule,
            classifier_cls,
            classifier_args,
            optimizer_cls,
            optimizer_args,
            scheduler_cls,
            scheduler_args,
            *args,
            **kwargs,
        )

        self.model = classifier_cls(**classifier_args)

        self.start_epoch = start_epoch
        self.alpha = alpha
        self.sigma = sigma
        self.momentum = momentum
        self.lamd_ce = lambd_ce
        self.lamd_h = lambd_h

        N = datamodule.num_train_samples
        C = datamodule.num_classes

        # Variable definition
        self.s_prev_confidence = torch.ones(N).to(self.device) * 1 / N
        self.w_prev_confidence = torch.ones(N).to(self.device) * 1 / N
        self.ws_prev_confidence = torch.ones(N).to(self.device) * 1 / N

        self.w_probs = torch.zeros(N, C).to(self.device)
        self.s_probs = torch.zeros(N, C).to(self.device)
        self.labels = torch.ones(N).long().to(self.device)
        self.weak_labels = self.labels.detach().clone()

        self.clean_flags = torch.zeros(N).bool().to(self.device)
        self.hard_flags = torch.zeros(N).bool().to(self.device)
        self.correction_flags = torch.zeros(N).bool().to(self.device)
        self.weak_flags = torch.zeros(N).bool().to(self.device)
        self.w_selected_flags = torch.zeros(N).bool().to(self.device)
        self.s_selected_flags = torch.zeros(N).bool().to(self.device)
        self.selected_flags = torch.zeros(N).bool().to(self.device)
        self.class_weight = torch.ones(C).to(self.device)

        # Loss function definition
        self.GCE_loss = GCELoss(num_classes=C, gpu="0")
        self.mixup_loss = Mixup(gpu="0", num_classes=C, alpha=alpha)
        self.criterion = nn.CrossEntropyLoss()

        # metrics
        self.train_acc = torchmetrics.Accuracy(
            num_classes=C, top_k=1, task="multiclass", average="micro"
        )
        self.val_acc = torchmetrics.Accuracy(
            num_classes=C, top_k=1, task="multiclass", average="micro"
        )
        self.test_acc = torchmetrics.Accuracy(
            num_classes=C, top_k=1, task="multiclass", average="micro"
        )

    def on_train_start(self):
        self.get_labels(self.datamodule.train_dataloader()[0])

    def training_step(self, batch, batch_idx):
        w_images, s_images, targets, indexes = batch[0]
        device = w_images.device

        w_imgs = torch.tensor(w_images).to(self.device)
        s_imgs = torch.tensor(s_images).to(self.device)
        targets = torch.tensor(targets).to(self.device)

        all_inputs = torch.cat([w_imgs, s_imgs], dim=0)
        bs = w_imgs.shape[0]
        logits = self.model(all_inputs)
        w_logits = logits[:bs]
        s_logits = logits[bs:]

        if self.current_epoch <= self.start_epoch:
            # ----- warmup ----- #
            loss_sup = self.criterion(w_logits, targets) + self.criterion(
                s_logits, targets
            )

            with torch.no_grad():
                w_prob = F.softmax(w_logits, dim=1)
                self.w_probs[indexes.cpu()] = w_prob.cpu()
                s_prob = F.softmax(s_logits, dim=1)
                self.s_probs[indexes.cpu()] = s_prob.cpu()

        else:
            # ----- actual training ----- #
            loss_sup = torch.tensor(0).float().to(self.device)

            b_clean_flags = self.clean_flags[indexes.cpu()]
            clean_num = b_clean_flags.sum()
            b_hard_flags = self.hard_flags[indexes.cpu()]
            hard_num = b_hard_flags.sum()

            batch_size = len(w_imgs)

            if clean_num:
                clean_loss_sup = self.criterion(
                    w_logits[b_clean_flags], targets[b_clean_flags]
                ) + self.criterion(s_logits[b_clean_flags], targets[b_clean_flags])
                loss_sup += clean_loss_sup * self.lamd_ce * (clean_num / batch_size)
            if hard_num:
                hard_loss_sup = self.GCE_loss(
                    w_logits[b_hard_flags], targets[b_hard_flags]
                ) + self.GCE_loss(s_logits[b_hard_flags], targets[b_hard_flags])
                loss_sup += hard_loss_sup * self.lamd_h * (hard_num / batch_size)

            # Mixup
            weak_labels = self.weak_labels[indexes.cpu()].to(device)
            weak_flag = self.weak_flags[indexes.cpu()].to(device)
            weak_num = weak_flag.sum()

            if weak_num:
                mixup_loss = self.mixup_loss(
                    w_imgs[weak_flag], weak_labels[weak_flag], self.model
                )
                mixup_loss += self.mixup_loss(
                    s_imgs[weak_flag], weak_labels[weak_flag], self.model
                )
                loss_sup += mixup_loss

            with torch.no_grad():
                w_prob = F.softmax(w_logits, dim=1)
                self.w_probs[indexes.cpu()] = w_prob.cpu()
                s_prob = F.softmax(s_logits, dim=1)
                self.s_probs[indexes.cpu()] = s_prob.cpu()

        self.log("train_loss", loss_sup, prog_bar=True)
        self.log(
            "train_acc", self.train_acc(w_logits, targets), on_epoch=True, on_step=False
        )

        return loss_sup

    def on_train_epoch_end(self):
        ws_probs = (self.w_probs + self.s_probs) / 2
        w_prob_max, w_label = torch.max(self.w_probs, dim=1)
        s_prob_max, s_label = torch.max(self.s_probs, dim=1)
        ws_prob_max, ws_label = torch.max(ws_probs, dim=1)

        ###############Selection###############
        w_mask = (
            self.w_probs[self.labels >= 0, self.labels]
            > self.w_prev_confidence[self.labels >= 0]
        )
        s_mask = (
            self.s_probs[self.labels >= 0, self.labels]
            > self.s_prev_confidence[self.labels >= 0]
        )
        self.clean_flags = w_mask & s_mask
        self.selected_flags = w_mask + s_mask
        self.w_selected_flags = w_mask & (~self.clean_flags)  # H_w
        self.s_selected_flags = s_mask & (~self.clean_flags)  # H_s
        self.hard_flags = self.w_selected_flags + self.s_selected_flags  # H
        #######################################

        ###############Correction##############
        ws_threshold = (
            self.w_prev_confidence + self.s_prev_confidence
        ) / 2 + self.sigma
        ws_threshold = torch.min(ws_threshold, torch.tensor(0.99))
        self.correction_flags = ws_prob_max > ws_threshold
        self.correction_flags = self.correction_flags & (
            ~self.selected_flags
        )  # P-(C+H)
        #######################################

        ###############Mix set###############
        self.weak_flags = self.correction_flags + self.selected_flags
        self.weak_labels[self.selected_flags] = self.labels[self.selected_flags]
        self.weak_labels[self.correction_flags] = ws_label[self.correction_flags]
        #######################################

        self.w_prev_confidence = (
            self.momentum * self.w_prev_confidence + (1 - self.momentum) * w_prob_max
        )

        self.s_prev_confidence = (
            self.momentum * self.s_prev_confidence + (1 - self.momentum) * s_prob_max
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        self.log("val_acc", self.val_acc(y_pred, y), on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        self.test_acc(y_pred, y)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_args)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def get_labels(self, train_loader):
        print("Loading labels......")
        pbar = tqdm(train_loader)
        for _, _, targets, indexes in pbar:
            self.labels[indexes] = targets
        self.weak_labels = self.labels.detach().clone()
        print("The labels are loaded!")
