# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from metrics import turbine_scores

__all__ = [ "SCORELoss", "FilterMSELoss", "MSELoss", "HuberLoss", "MAELoss", "SmoothMSELoss"]

class SCORELoss(nn.Module):
    def __init__(self, **kwargs):
        super(SCORELoss, self).__init__()

    def forward(self, pred, gold, raw, col_names):
        gold = gold.to(pred.device)
        all_mae, all_rmse = 0, 0
        for i in range(pred.size(1)):
            prediction = pred[:, i, :]
            gt = gold[:, i, :]
            _mae = F.l1_loss(prediction, gt)
            _rmse = torch.sqrt(F.mse_loss(prediction, gt))
            all_mae += _mae
            all_rmse += _rmse
        score_loss = (all_mae + all_rmse)/2
        return score_loss.to(pred.device)

class FilterMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(FilterMSELoss, self).__init__()

    def forward(self, pred, gold, raw, col_names):
        # Remove bad input
        cond1 = raw[:, :, :, col_names["Patv"]] < 0

        cond2 = raw[:, :, :, col_names["Pab1"]] > 89
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab2"]] > 89)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab3"]] > 89)

        cond2 = torch.logical_or(cond2,
                                  raw[:, :, :, col_names["Wdir"]] < -180)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Wdir"]] > 180)
        cond2 = torch.logical_or(cond2,
                                  raw[:, :, :, col_names["Ndir"]] < -720)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Ndir"]] > 720)
        cond2 = torch.logical_or(cond2, cond1)

        cond3 = raw[:, :, :, col_names["Patv"]] == 0
        cond3 = torch.logical_and(cond3,
                                   raw[:, :, :, col_names["Wspd"]] > 2.5)
        cond3 = torch.logical_or(cond3, cond2)

        cond = torch.logical_not(cond3)
        cond = cond.type(torch.float32).to(pred.device)
        gold = gold.to(pred.device)

        _mse = torch.mean(F.mse_loss(pred, gold, reduction='none') * cond, axis=2)
        _mae = torch.mean(F.l1_loss(pred, gold, reduction='none') * cond, axis=2)

        _mse, _ = torch.sort(torch.sum(_mse, axis=1), descending=True)
        _mae, _ = torch.sort(torch.sum(_mae, axis=1), descending=True)

        top_l = int(len(pred) * 1)
        # print(top_l, 'loss shape', _mse.shape, _mae.shape, _mse, '*'*10, _mae)
        return torch.mean(_mse[:top_l]) + torch.mean(_mae[:top_l])


class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()

    def forward(self, pred, gold, raw, col_names):
        return F.mse_loss(pred, gold)


class MAELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MAELoss, self).__init__()

    def forward(self, pred, gold, raw, col_names):
        loss = F.l1_loss(pred, gold)
        return loss


class HuberLoss(nn.Module):
    def __init__(self, delta=5, **kwargs):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, gold, raw, col_names):
        loss = F.smooth_l1_loss(pred, gold, reduction='mean', delta=self.delta)
        return loss


class SmoothMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(SmoothMSELoss, self).__init__()
        self.smooth_win = kwargs["smooth_win"]

    def forward(self, pred, gold, raw, col_names):
        gold = F.avg_pool1d(
            gold, self.smooth_win, stride=1, padding="SAME", exclusive=False)
        loss = F.mse_loss(pred, gold)
        return loss
