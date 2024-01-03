import albumentations as A
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from models.model import *
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from val import evaluate
from tqdm import tqdm

PALETTE = [[0, 0, 0], [255, 255, 255]]


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


def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    ground_truth = ground_truth.float()
    predictions = predictions.float()
    ground_truth = ground_truth.view(-1)
    predictions = predictions.view(-1)

    intersection = torch.sum(predictions * ground_truth)
    union = torch.sum(predictions) + torch.sum(ground_truth)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1 - dice


class PolypDB(Dataset):
    def __init__(self, root: str, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.n_classes = 2
        self.ignore_label = -1

        img_path = Path(root) / "images"
        self.files = list(img_path.glob("*.jpg"))

        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} images.")

    @staticmethod
    def convert_to_mask(mask):
        h, w = mask.shape[:2]
        seg_mask = np.zeros((h, w, len(PALETTE)))
        for i, label in enumerate(PALETTE):
            seg_mask[:, :, i] = np.all(mask == label, axis=-1)
        return seg_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace("images", "masks")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(lbl_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.convert_to_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)

            image = transformed["image"]
            mask = transformed["mask"]
            return image.float(), mask.argmax(dim=2).long()

        else:
            return image.float(), mask.argmax(dim=2).long()


def create_dataloaders(dir, image_size, batch_size, num_workers=os.cpu_count()):
    if isinstance(image_size, int):
        image_size = [image_size, image_size]

    transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dataset = PolypDB(root=dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return dataloader, dataset


def main(save_dir, train_loader, val_loader):
    start = time.time()
    best_mIoU = 0.0
    device = torch.device("cuda")
    in_channels = 17
    model = DUCK_Net(in_channels)
    epochs, lr = 20, 0.001

    model = model.to(device)

    writer = SummaryWriter(str(save_dir / "logs"))
    loss_record = AvgMeter()
    size_rates = [0.75, 1, 1.25]

    iters_per_epoch = len(train_loader.dataset) // 8

    optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(
            enumerate(train_loader),
            total=iters_per_epoch,
            desc=f"Epoch: [{epoch}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}",
        )

        for iter, (img, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)

            img = img.to(device)
            lbl = lbl.to(device)
            
            logits = model(img)
            loss = dice_metric_loss(logits, lbl)
            loss.backward()

            loss_record.update(loss.data, 8)

            optimizer.step()

            train_loss = loss.data

            pbar.set_description(
                f"Epoch: [{epoch}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {optimizer.param_groups[0]['lr']:.8f} Loss: {loss_record.show()}"
            )

        train_loss /= iter + 1
        writer.add_scalar("train/loss", train_loss, epoch)
        torch.cuda.empty_cache()

        if epoch % 10 == 0 or epoch == epochs:
            miou = evaluate(model, val_loader, device, "Training")[0]

            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.state_dict(), save_dir / "best.pth")
            torch.save(model.state_dict(), save_dir / f"checkpoint{epoch+1}.pth")

            print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")

    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)
    print(time.strftime("%H:%M:%S", end))


if __name__ == "__main__":
    ds = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir-SEG"]
    # ds = ["PolypGen"]
    for _ds in ds:
        print(_ds)
        save_path = f"results/DUCK-Net/{_ds}"
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)

        train_path = f"data/Datasets/{_ds}/train/"
        train_loader, dataset = create_dataloaders(train_path, [352, 352], 8, True)
        val_path = f"data/Datasets/{_ds}/validation/"
        val_loader, dataset = create_dataloaders(val_path, [352, 352], 1, False)

        main(save_dir, train_loader, val_loader)
