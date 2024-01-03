import torch
from torch import Tensor
from typing import Tuple

import numpy as np

#-----------------------------------------------------#
#            Calculate : Confusion Matrix             #
#-----------------------------------------------------#


def calc_ConfusionMatrix(truth, pred, c=1, dtype=np.int64, **kwargs):
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    tp = np.logical_and(pd, gt).sum()
    tn = np.logical_and(not_pd, not_gt).sum()
    fp = np.logical_and(pd, not_gt).sum()
    fn = np.logical_and(not_pd, gt).sum()
    tp = tp.astype(dtype)
    tn = tn.astype(dtype)
    fp = fp.astype(dtype)
    fn = fn.astype(dtype)
    return tp, tn, fp, fn

#-----------------------------------------------------#
#              Calculate : True Positive              #
#-----------------------------------------------------#


def calc_TruePositive(truth, pred, c=1, **kwargs):
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    tp = np.logical_and(pd, gt).sum()
    return tp

#-----------------------------------------------------#
#              Calculate : True Negative              #
#-----------------------------------------------------#


def calc_TrueNegative(truth, pred, c=1, **kwargs):
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    tn = np.logical_and(not_pd, not_gt).sum()
    return tn

#-----------------------------------------------------#
#              Calculate : False Positive             #
#-----------------------------------------------------#


def calc_FalsePositive(truth, pred, c=1, **kwargs):
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    fp = np.logical_and(pd, not_gt).sum()
    return fp

#-----------------------------------------------------#
#              Calculate : False Negative             #
#-----------------------------------------------------#


def calc_FalseNegative(truth, pred, c=1, **kwargs):
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    fn = np.logical_and(not_pd, gt).sum()
    return fn

#-----------------------------------------------------#
#              Calculate : DSC Enhanced               #
#-----------------------------------------------------#


def calc_DSC_Enhanced(truth, pred, c=1, **kwargs):
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    if gt.sum() == 0 and pd.sum() == 0:
        dice = 1.0
    elif (pd.sum() + gt.sum()) != 0:
        dice = 2*np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum())
    else:
        dice = 0.0
    return dice

#-----------------------------------------------------#
#              Calculate : DSC via Sets               #
#-----------------------------------------------------#


def calc_DSC_Sets(truth, pred, c=1, **kwargs):
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    if (pd.sum() + gt.sum()) != 0:
        dice = 2*np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum())
    else:
        dice = 0.0
    return dice

#-----------------------------------------------------#
#             Calculate : DSC via ConfMat             #
#-----------------------------------------------------#


def calc_DSC_CM(truth, pred, c=1, **kwargs):
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
    if (2*tp + fp + fn) != 0:
        dice = 2*tp / (2*tp + fp + fn)
    else:
        dice = 0.0
    return dice


import numpy as np

def calc_ConfusionMatrix(truth, pred, class_id, dtype=np.int64):
    gt = np.equal(truth, class_id)
    pd = np.equal(pred, class_id)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    tp = np.logical_and(pd, gt).sum()
    tn = np.logical_and(not_pd, not_gt).sum()
    fp = np.logical_and(pd, not_gt).sum()
    fn = np.logical_and(not_pd, gt).sum()
    tp = tp.astype(dtype)
    tn = tn.astype(dtype)
    fp = fp.astype(dtype)
    fn = fn.astype(dtype)
    return tp, tn, fp, fn

def calc_DSC_CM(truth, pred, class_id):
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, class_id)
    if (2*tp + fp + fn) != 0:
        dice = 2*tp / (2*tp + fp + fn)
    else:
        dice = 0.0
    return dice

def calc_F1_Score(tp, fp, fn):
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1_score, precision, recall

class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = [0] * self.num_classes
        self.tn = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.fn = [0] * self.num_classes

    def update(self, pred, label):
        pred = pred.flatten().cpu().numpy()
        label = label.flatten().cpu().numpy()

        for class_id in range(self.num_classes):
            tp, tn, fp, fn = calc_ConfusionMatrix(label, pred, class_id)
            self.tp[class_id] += tp
            self.tn[class_id] += tn
            self.fp[class_id] += fp
            self.fn[class_id] += fn

    def compute_iou(self):
        iou = [0] * self.num_classes
        for class_id in range(self.num_classes):
            iou[class_id] = self.tp[class_id] / (self.tp[class_id] + self.fp[class_id] + self.fn[class_id] + 1e-7)
        return iou

    def compute_dice(self):
        dice = [0] * self.num_classes
        for class_id in range(self.num_classes):
            dice[class_id] = 2 * self.tp[class_id] / (2 * self.tp[class_id] + self.fp[class_id] + self.fn[class_id] + 1e-7)
        return dice


    def compute_mean_iou(self):
        iou = self.compute_iou()
        mean_iou = np.mean(iou)
        return mean_iou

    def compute_mean_dice(self):
        dice = self.compute_dice()
        mean_dice = np.mean(dice)
        return mean_dice
    
    def compute_f1_score(self):
        f1_scores = [0] * self.num_classes
        precisions = [0] * self.num_classes
        recalls = [0] * self.num_classes
        for i in range(self.num_classes):
            f1_scores[i], precisions[i], recalls[i] = calc_F1_Score(self.tp[i], self.fp[i], self.fn[i])
        return f1_scores, precisions, recalls

    def compute_mean_f1_score(self):
        f1_scores, precisions, recalls = self.compute_f1_score()
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1_score = np.mean(f1_scores)
        return mean_f1_score, mean_precision, mean_recall

    def compute_single_iou(self, label, pred):
        tps = [0] * self.num_classes
        tns = [0] * self.num_classes
        fps = [0] * self.num_classes
        fns = [0] * self.num_classes
        pred = pred.flatten().cpu().numpy()
        label = label.flatten().cpu().numpy()

        for class_id in range(self.num_classes):
            tp, tn, fp, fn = calc_ConfusionMatrix(label, pred, class_id)
            tps[class_id] += tp
            tns[class_id] += tn
            fps[class_id] += fp
            fns[class_id] += fn
        iou = [0] * self.num_classes
        for class_id in range(self.num_classes):
            iou[class_id] = tps[class_id] / (tps[class_id] + fps[class_id] + fns[class_id] + 1e-7)
        return np.mean(iou)



