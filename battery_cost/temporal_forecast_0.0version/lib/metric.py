import torch
import torch.nn as nn


class TweedieLoss(nn.Module):
    """
    Tweedie Loss
    :param p: Tweedie power
    """
    def __init__(self, p=1.5):
        super(TweedieLoss, self).__init__()
        self.p = p

    def forward(self, x, y):
        # x: predict
        # y: label
        loss = - y * torch.pow(x + 1e-8, 1 - self.p) / (1 - self.p) + torch.pow(x, 2 - self.p) / (2 - self.p)
        return torch.mean(loss)


def MAPE(y_true, y_pred):
    cnt = 0
    mape = 0.0
    for i in range(len(y_true)):
        if y_true[i] >= 10:
            cnt += 1
            mape += abs(y_pred[i] - y_true[i]) / y_true[i]
    if cnt == 0:
        return 0
    return (mape / cnt)


def WMAPE(y_true, y_pred):
    sum_abs = 0.0
    sum_y = 0.0
    for i in range(len(y_true)):
        sum_abs += abs(y_true[i] - y_pred[i])
        sum_y += y_true[i]
    res = sum_abs / sum_y
    return res


def MAE(y_true, y_pred):
    cnt0 = 0
    mae0 = 0.0

    cnt1 = 0
    mae1 = 0.0

    cnt2 = 0
    mae2 = 0.0

    cnt3 = 0
    mae3 = 0.0
    for i in range(len(y_true)):
        if y_true[i] == 0:
            cnt0 += 1
            mae0 += abs(y_pred[i] - y_true[i])
        elif y_true[i] > 0 and y_true[i] <= 10:
            cnt1 += 1
            mae1 += abs(y_pred[i] - y_true[i])
        elif y_true[i] > 10 and y_true[i] <= 20:
            cnt2 += 1
            mae2 += abs(y_pred[i] - y_true[i])
        else:
            cnt3 += 1
            mae3 += abs(y_pred[i] - y_true[i])
    try:
        a = mae0 / cnt0
        a = a
    except:
        a = 0.0
    try:
        b = (mae1 / cnt1)
    except:
        b = 0.0
    try:
        c = (mae2 / cnt2)
    except:
        c = 0.0
    try:
        d = (mae3 / cnt3)
    except:
        d = 0.0

    return a, b, c, d

import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np
from sklearn import metrics

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def RRSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))

def CORR_torch(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(dim=1)
        true = true.transpose(1, 2).unsqueeze(dim=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(dim=dims)
    true_mean = true.mean(dim=dims)
    pred_std = pred.std(dim=dims)
    true_std = true.std(dim=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(dim=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation

def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true))) * 100



def All_Metrics(pred, true, mask1):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == torch.Tensor:
        mae  = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
#        mape = MAPE_torch(pred, true, mask2)
        rrse = RRSE_torch(pred, true, mask1)
        #corr = CORR_torch(pred, true, mask1)
        #pnbi = PNBI_torch(pred, true, mask1)
        #opnbi = oPNBI_torch(pred, true, mask2)
    else:
        raise TypeError
    return mae, rmse, rrse



