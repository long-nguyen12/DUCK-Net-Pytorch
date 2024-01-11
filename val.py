import torch
# import argparse
# import yaml
# from pathlib import Path
from tqdm import tqdm
# from models.models import *
# from models.datasets import *
from metrics import Metrics
# from models.utils.utils import setup_cudnn
import os
# import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from models.datasets.polyp import PolypDB
from torchvision import utils

PALETTE = [[0, 0, 0], [255, 255, 255]]


@torch.no_grad()
def evaluate(model, dataloader, device, folder):
    print("Evaluating...")
    model.eval()
    metrics = Metrics(2, -1, device)

    save_preds = "./result/preds/" + "/" + folder + "/"
    save_labels = "./result/labels/" + "/" + folder + "/"
    save_images = "./result/images/" + "/" + folder + "/"

    if not os.path.exists(save_labels):
        os.makedirs(save_labels)
    if not os.path.exists(save_preds):
        os.makedirs(save_preds)
    if not os.path.exists(save_images):
        os.makedirs(save_images)

    count = 0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        # preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        save_path = save_preds + str(count) + "_best.png" 
        utils.save_image(preds, save_path)

        metrics.update(preds, labels)
        count += 1

    miou = metrics.compute_mean_iou()
    mdice = metrics.compute_mean_dice()
    f1_score, mprecision, mrecall = metrics.compute_mean_f1_score()

    return miou, mdice, f1_score, mprecision, mrecall


# def create_dataloaders(dir, image_size):
#     if isinstance(image_size, int):
#         image_size = [image_size, image_size]

#     transform = A.Compose(
#         [
#             A.Resize(height=image_size[0], width=image_size[1]),
#             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2(),
#         ]
#     )

#     dataset = PolypDB(root=dir, transform=transform)
#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=1, num_workers=1, pin_memory=True
#     )

#     return dataloader, dataset


# def main(cfg, dataloader, dataset, _dataset):
#     device = torch.device(cfg["DEVICE"])

#     eval_cfg = cfg["EVAL"]

#     model_path = Path(eval_cfg["MODEL_PATH"])

#     model = eval(cfg["MODEL"]["NAME"])(cfg["MODEL"]["BACKBONE"], 2)
#     model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
#     model = model.to(device)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Number of parameters: {total_params}")
#     miou, mdice, f1_score, mprecision, mrecall = evaluate(
#         model, dataloader, device, _dataset
#     )

#     return miou, mdice, f1_score, mprecision, mrecall


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cfg", type=str, default="configs/custom.yaml")
#     args = parser.parse_args()

#     with open(args.cfg) as f:
#         cfg = yaml.load(f, Loader=yaml.SafeLoader)

#     setup_cudnn()
#     ds = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]

#     for _dataset in ds:
#         ious = []
#         dices = []
#         f1_scores = []
#         precisions = []
#         recalls = []
#         print(_dataset)
#         dataloader, dataset = create_dataloaders(
#             "data/data/TestDataset/" + _dataset, [352, 352]
#         )
#         for i in range(5):
#             iou, dice, f1_score, mprecision, mrecall = main(
#                 cfg, dataloader, dataset, _dataset
#             )
#             ious.append(iou)
#             dices.append(dice)
#             f1_scores.append(f1_score)
#             precisions.append(mprecision)
#             recalls.append(mrecall)
#             print(f"Mean IoU: {iou}, mean Dice: {dice}")

#         ious = np.array(ious)
#         dices = np.array(dices)
#         f1_scores = np.array(f1_scores)
#         precisions = np.array(precisions)
#         recalls = np.array(recalls)
#         print(
#             f"Mean IoU: {np.mean(ious)}, mean Dice: {np.mean(dices)}, mean F1-Score: {round(np.mean(f1_scores), 3)}, mean Precision: {round(np.mean(precisions), 3)}, mean Recall: {round(np.mean(recalls), 3)}"
#         )
