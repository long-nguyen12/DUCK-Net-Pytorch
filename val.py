import argparse
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from models.model import DUCK_Net


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
        mask = mask.astype("float32")
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


def get_scores(gts, prs):
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    for gt, pr in zip(gts, prs):
        mean_precision += precision_m(gt, pr)
        mean_recall += recall_m(gt, pr)
        mean_iou += iou_m(gt, pr)
        mean_dice += dice_m(gt, pr)

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)

    print(
        "scores: dice={}, miou={}, precision={}, recall={}".format(
            mean_dice, mean_iou, mean_precision, mean_recall
        )
    )

    return (mean_iou, mean_dice, mean_precision, mean_recall)


@torch.no_grad()
def inference(model, data_path):
    print("#" * 20)
    torch.cuda.empty_cache()
    model.eval()
    device = torch.device("cuda")
    X_test = glob("{}/images/*".format(data_path))
    X_test.sort()
    y_test = glob("{}/masks/*".format(data_path))
    y_test.sort()

    transform = A.Compose(
        [
            A.Resize(height=352, width=352),
            ToTensorV2(),
        ]
    )

    test_dataset = Dataset(X_test, y_test, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True
    )

    gts = []
    prs = []
    for i, pack in enumerate(test_loader, start=1):
        image, gt = pack

        image = image.to(device)
        gt = gt.to(device)
        res = model(image)
        pr = res
        pr = (pr > 0.5).float()
        gts.append(gt)
        prs.append(pr)
    return get_scores(gts, prs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="b3")
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument(
        "--test_path", type=str, default="./data/dataset", help="path to dataset"
    )
    parser.add_argument(
        "--init_trainsize", type=str, default=352, help="path to dataset"
    )
    args = parser.parse_args()
    ds = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir-SEG"]
    for _ds in ds:

        device = torch.device("cuda")
        in_channels = 17
        model = DUCK_Net(in_channels)
        model = model.to(device)

        checkpoint = torch.load(
            f"snapshots/mscan-base/{_ds}/best.pth", map_location="cpu"
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)

        # ds = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
        # for _ds in ds:
        print(_ds)
        data_path = args.test_path + "/" + _ds + "/test/"
        inference(model, data_path, args)
