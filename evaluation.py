import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def cal_auc(mask, predict):
    auc_score = roc_auc_score(mask.flatten().astype('int'), predict.flatten())
    return auc_score

def cal_precision(mask, predict):
    TP = ((mask == 1) & (predict == 1))
    FP = ((mask == 1) & (predict == 0))
    return np.sum(TP, dtype=np.float32) / ((np.sum(TP, dtype=np.float32) + np.sum(FP, dtype=np.float32)) + 1e-6)

def cal_recall(mask, predict):
    TP = ((mask == 1) & (predict == 1))
    FN = ((mask == 0) & (predict == 1))
    return np.sum(TP, dtype=np.float32) / ((np.sum(TP, dtype=np.float32) + np.sum(FN, dtype=np.float32)) + 1e-6)

def cal_f1(mask, predict):
    precision = cal_precision(mask, predict)
    recall = cal_recall(mask, predict)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1

