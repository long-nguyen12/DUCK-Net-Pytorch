import os
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from datetime import datetime

from models.model import *
from models.model import DUCK_Net
from val import inference
from glob import glob


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(
            torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0) :])
        )


epsilon = 1e-7


def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall


def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + epsilon))


def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall * precision / (recall + precision - recall * precision + epsilon)


def dice_metric_loss(ground_truth, predictions):
    dice = dice_m(ground_truth, predictions)
    return 1 - dice


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        mask = mask[:, :, np.newaxis]
        mask = mask.astype("float32") / 255
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352))

        image = image.float()

        mask = mask.permute(2, 0, 1)

        return image, mask

if __name__ == "__main__":
    ds = ["CVC-ColonDB", "CVC-ClinicDB", "ETIS-LaribPolypDB", "Kvasir-SEG"]
    for _ds in ds:
        print(_ds)
        save_path = f"results/DUCK-Net/{_ds}"
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)

        train_img_paths = []
        train_mask_paths = []
        train_img_paths = glob("data/dataset/{}/train/images/*".format(_ds))
        train_mask_paths = glob("data/dataset/{}/train/masks/*".format(_ds))
        train_img_paths.sort()
        train_mask_paths.sort()

        transform = A.Compose(
            [
                A.Resize(height=352, width=352),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.ColorJitter(
                    brightness=(0.6, 1.6),
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.01,
                    always_apply=True,
                ),
                A.Affine(
                    scale=(0.5, 1.5),
                    translate_percent=(-0.125, 0.125),
                    rotate=(-180, 180),
                    shear=(-22.5, 22),
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )

        train_dataset = Dataset(train_img_paths, train_mask_paths, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        start = time.time()
        best_mIoU = 0.0
        device = torch.device("cuda")
        in_channels = 17
        model = DUCK_Net(in_channels)
        model = model.to(device)

        epochs, lr = 600, 0.0001

        loss_record = AvgMeter()
        dice, iou = AvgMeter(), AvgMeter()

        iters_per_epoch = len(train_loader)
        total_step = len(train_loader)

        params = model.parameters()
        optimizer = optim.RMSprop(params, lr=lr)

        for epoch in range(1, epochs + 1):
            model.train()
            with torch.autograd.set_detect_anomaly(True):
                for i, pack in enumerate(train_loader, start=1):
                    optimizer.zero_grad()
                    img, lbl = pack

                    img = img.to(device)
                    lbl = lbl.to(device)

                    logits = model(img)

                    loss = dice_metric_loss(logits, lbl)
                    dice_score = dice_m(logits, lbl)
                    iou_score = iou_m(logits, lbl)

                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()

                    loss_record.update(loss.data, 4)
                    dice.update(dice_score.data, 4)
                    iou.update(iou_score.data, 4)

                    train_loss = loss.data


                print(
                    "{} Training Epoch [{:03d}/{:03d}], "
                    "[loss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}]".format(
                        datetime.now(),
                        epoch,
                        epochs,
                        loss_record.show(),
                        dice.show(),
                        iou.show(),
                    ),
                    flush=True,
                )

                if epoch % 10 == 0 or epoch == epochs:
                    miou, mean_dice, mean_precision, mean_recall = inference(
                        model, f"data/dataset/{_ds}/test/"
                    )

                    if miou > best_mIoU:
                        best_mIoU = miou
                        torch.save(model.state_dict(), save_dir / "best.pth")
                    torch.save(model.state_dict(), save_dir / f"last.pth")

                    print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")

        end = time.gmtime(time.time() - start)
        print(time.strftime("%H:%M:%S", end))

