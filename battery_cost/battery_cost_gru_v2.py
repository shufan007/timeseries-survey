import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score,f1_score,mean_absolute_error,mean_squared_error
import gc
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import random
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str)
parser.add_argument('--test_dir', type=str)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--loss', type=str, default="tweedie")
parser.add_argument('--ver', type=str)
parser.add_argument('--label_index', type=int, default=3)

args = parser.parse_args()

train_len = 0
test_len = 0

import os

for file in os.listdir(args.train_dir):
    if 'part' not in file:
        continue
    train_len += 1

for file in os.listdir(args.test_dir):
    if 'part' not in file:
        continue
    test_len += 1
    
print(train_len, test_len)


def mape(y_true, y_pred):
    cnt = 0
    mape = 0.0
    for i in range(len(y_true)):
        if y_true[i] >= 10:
            cnt += 1
            mape += abs(y_pred[i] - y_true[i]) / y_true[i]
    if cnt == 0:
        return 0
    return (mape / cnt)

def wmape(y_true, y_pred):
    sum_abs = 0.0
    sum_y = 0.0
    for i in range(len(y_true)):
        sum_abs += abs(y_true[i] - y_pred[i])
        sum_y += y_true[i]
    return sum_abs / sum_y

def mae(y_true, y_pred):
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
    except:
        a = 0.0
    try:
        b = mae1 / cnt1
    except:
        b = 0.0
    try:
        c = mae2 / cnt2
    except:
        c = 0.0
    try:
        d = mae3 / cnt3
    except:
        d = 0.0
        
    return a, b, c, d


def encode_label(x):
    return np.log(x + 1)
def decode_label(x):
    return np.exp(x)

from torch.utils.data import Dataset, DataLoader

class SegData(Dataset):
    def __init__(self, file_name, input_len, label_index=-1):
        self.samples = torch.load(file_name)
        self.input_len = input_len
        self.label_index = label_index

    def __getitem__(self, idx):
        geo = self.samples[0][idx]
        version = self.samples[1][idx]
        raw_label = self.samples[2][idx][self.label_index]
        features = []
        for i in range(3, self.input_len):
            features.append(self.samples[i][idx])
        return geo, version, features, raw_label

    def __len__(self):
        return self.samples[1].shape[0]


def get_feature(f, use_gpu, feature_types):
    for i in range(feature_types):
        if use_gpu == 1:
            if i == 0 or i == 3 or i == 6:
                f[i] = f[i].long().cuda()
            else:
                f[i] = f[i].float().cuda()
        else:
            if i == 0 or i == 3 or i == 6:
                f[i] = f[i].long()
            else:
                f[i] = f[i].float()
    return f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]

import math
import tqdm

class MultiHeadSelectAttention(nn.Module):
    def __init__(self, input_size, n_head, project_size, value_input, value_size):
        super(MultiHeadSelectAttention, self).__init__()
        self.n_head = n_head
        self.key_size = self.query_size = project_size // n_head
        self.trans_q = nn.Linear(input_size, project_size, bias=False)
        self.trans_k = nn.Linear(input_size, project_size, bias=False)
        self.trans_v = nn.Linear(value_input, value_size * n_head, bias=False)
        self.value_size = value_size
        self.sqrt_k = math.sqrt(self.key_size)
        self.layer_norm = nn.LayerNorm(value_size * n_head)
        
    def forward(self, query_input, key_input, value_input):
        batch_size, seq_len, _ = query_input.shape  
        query = self.trans_q(query_input).transpose(2, 1).reshape(batch_size * self.n_head, seq_len, -1)
        key = self.trans_k(key_input).reshape(batch_size * self.n_head, -1, 1)
        tmp_v = self.trans_v(value_input).transpose(2, 1).reshape(batch_size * self.n_head, seq_len, -1)
        # b * n * seq_len
        dot = torch.softmax(torch.bmm(query, key).squeeze(1) / self.sqrt_k, 1)
        value = dot.expand(batch_size * self.n_head, seq_len, self.value_size) * tmp_v
        skip_connect = value.view(batch_size, self.n_head, seq_len, -1).transpose(2,1).reshape(batch_size, seq_len, -1)
        res = self.layer_norm(skip_connect)
        return res

class TweedieLoss(nn.Module):
    """Tweedie Loss
    :param p: Tweedie power
    """
    def __init__(self, p=1.5):
        super(TweedieLoss, self).__init__()
        self.p = p

    def forward(self, x, y):
        # x: predict
        # y: label
        loss = - y * torch.pow(x+1e-8, 1 - self.p) / (1 - self.p) + torch.pow(x, 2 - self.p) / (2 - self.p)
        return torch.mean(loss)

import math
import tqdm

class GruModel(nn.Module):
    def __init__(self, cate_shape, cate_size, seq_nume_size, flat_nume_size, cat_hidden, seq_hidden, static_size, mha_attn_hidden, n_head, extra_flat_dim, pre_train=True):
        super(GruModel, self).__init__()
        self.cat_emb = nn.Embedding(cate_shape, cat_hidden, padding_idx=cate_shape-1)
        self.seq_hidden = seq_hidden
        self.pre_train = pre_train
        
        seq_in = cate_size * cat_hidden + seq_nume_size
        static_input_size = static_size + cate_size * cat_hidden
        gru_in = n_head * mha_attn_hidden + seq_in
        
        self.h_gru = nn.GRU(gru_in, seq_hidden, 1, batch_first=True)
        self.h_attn = MultiHeadSelectAttention(static_input_size, n_head, n_head * mha_attn_hidden, seq_in, mha_attn_hidden)
        self.d_gru = nn.GRU(gru_in, seq_hidden, 1, batch_first=True)
        self.d_attn = MultiHeadSelectAttention(static_input_size, n_head, n_head * mha_attn_hidden, seq_in, mha_attn_hidden)

        flat_in = cate_size * cat_hidden + flat_nume_size + extra_flat_dim + seq_hidden * 2
        
        self.fc = nn.Sequential(
            nn.Linear(flat_in, 256)
            , nn.BatchNorm1d(256)
            , nn.Tanh()
            , nn.Linear(256, 256)
            , nn.BatchNorm1d(256)
            , nn.Tanh()
            , nn.Linear(256, 128)
            , nn.BatchNorm1d(128)
            , nn.Tanh()
            , nn.Linear(128, 128)
            , nn.BatchNorm1d(128)
            , nn.Tanh()
            , nn.Linear(128, 64)
            , nn.BatchNorm1d(64)
            , nn.Tanh()
            , nn.Linear(64, 64)
            , nn.BatchNorm1d(64)
            , nn.Tanh()
            , nn.Dropout(p=0.5)
            , nn.Linear(64, 1)
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.pre_train:
            pretrain_dict = torch.load(PRETRAIN_MODEL_FILE)
            model_dict = self.state_dict()
            for k in pretrain_dict.keys():
                if k == 'fc.0.weight':
                    for i in range(len(model_dict[k])):
                        model_dict[k][i] = torch.cat([pretrain_dict[k][i][:188], self.ZEROS_TENSOR, pretrain_dict[k][i][188:]])
                else:
                    model_dict[k] = pretrain_dict[k]
            self.load_state_dict(model_dict)
            print('='*50)
            print('Pretrain Model Load Success!')
            print('='*50)
        else:
            std = 1.0 / math.sqrt(self.seq_hidden)
            for w in self.parameters():
                w.data.uniform_(-std, std)
        
    def forward(self, batch_flat_int, batch_flat_float, batch_flat_dense,
                batch_h_int, batch_h_float, batch_h_label,
                batch_seq_int, batch_seq_float, batch_seq_label):

        batch_size = batch_flat_int.shape[0]
        h_seq_len = batch_h_int.shape[1]
        seq_len = batch_seq_int.shape[1]
        
        flat_emb = self.cat_emb(batch_flat_int).view(batch_size, -1)
        h_seq_emb = self.cat_emb(batch_h_int).view(batch_size, h_seq_len, -1)
        seq_emb = self.cat_emb(batch_seq_int).view(batch_size, seq_len, -1)
        
        flat_static = torch.cat([flat_emb, batch_flat_float[:, :24]], axis=-1)
        h_seq_static = torch.cat([h_seq_emb, batch_h_float[:,:, :24]], axis=-1)
        seq_static = torch.cat([seq_emb, batch_seq_float[:, :, :24]], axis=-1)

        h_seq_value = torch.cat([h_seq_emb, batch_h_float, batch_h_label], axis=-1)
        seq_value = torch.cat([seq_emb, batch_seq_float, batch_seq_label], axis=-1)
        
        h_seq_attn_out = self.h_attn(h_seq_static, flat_static, h_seq_value)
        seq_attn_out = self.d_attn(seq_static, flat_static, seq_value)
        h_gru_in = torch.cat([h_seq_value, h_seq_attn_out], axis=-1)
        d_gru_in = torch.cat([seq_value, seq_attn_out], axis=-1)
        h_out, h_hn = self.h_gru(h_gru_in)
        d_out, d_hn = self.d_gru(d_gru_in)
        
        flat_in = torch.cat([flat_emb, batch_flat_float, batch_flat_dense, h_hn.squeeze(0), d_hn.squeeze(0)], axis=1)
        res = self.fc(flat_in)
        return res
    
loss_func = TweedieLoss()
if args.loss == "tweedie":
    loss_func = TweedieLoss(p=1.4)
elif args.loss == "mae":
    loss_func = nn.L1Loss()
elif args.loss == "poisson":
    loss_func = nn.PoissonNLLLoss(log_input=True, reduction='none', full=False)
print(loss_func)

def eval_result(batch_size, model, epoch):
    model.eval().cuda()
    test_loss = 0.
    test_cnt = 0
    test_pred = []
    test_label = []
    with torch.no_grad():
        for i in range(test_len):
            try:
                test_data = SegData(args.test_dir + "/part-%s" % i, 12, int(args.label_index))
            except:
                continue
            if len(test_data) == 0:
                continue
            myloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
            for idx, batch in enumerate(myloader):
                geo, version, features, raw_label = batch
                if args.label_index == 2:
                    raw_label /= 1000
                flat_seq_c, flat_seq_d, flat_d, hh_seq_c, hh_seq_d, hh_seq_label, seq_c, seq_d, seq_label = get_feature(f = features, use_gpu = 1, feature_types = 9)
                y = raw_label.reshape(-1, 1).float().cuda()
                result_val = model(flat_seq_c, flat_seq_d, flat_d,  hh_seq_c, hh_seq_d, hh_seq_label, seq_c[:, :, -1, :], seq_d[:, :, -1, :], seq_label[:, :, -1, :])
                now_pred = result_val.cpu().data.numpy()
                now_y = y.cpu().data.numpy()
            
                test_pred += now_pred.reshape(-1).tolist()
                test_label += now_y.reshape(-1).tolist()
                if args.loss == "poisson" or args.loss == "mae":
                    test_loss += (loss_func(result_val, y)).mean().data * len(y)
                elif args.loss == "tweedie":
                    test_loss += (loss_func(torch.exp(result_val), y)).mean().data * len(y)
                test_cnt += len(y)
        test_dataset_len = len(test_label)
        if args.loss == "poisson" or args.loss == "tweedie":
            test_mae = mean_absolute_error(test_label, decode_label(test_pred))
            test_mape = mape(test_label, decode_label(test_pred))
            test_wmape = wmape(test_label, decode_label(test_pred))
            mae_0, mae_0_10, mae_10_20, mae_20 = mae(test_label, decode_label(test_pred))
        elif args.loss == "mae":
            test_mae = mean_absolute_error(test_label, test_pred)
            test_mape = mape(test_label, test_pred)
            test_wmape = wmape(test_label, test_pred)
            mae_0, mae_0_10, mae_10_20, mae_20 = mae(test_label, test_pred)
        test_loss /= test_cnt
        print('test mae 0 = %.4f\ntest mae 0-10 = %.4f\ntest mae 10-20 = %.4f\ntest mae 20 = %.4f\n'
                  % (mae_0, mae_0_10, mae_10_20, mae_20))
        print('Test predict std is %.4f' % (np.std(np.array(test_pred))))
        print('Test Iteration %d, mae = %.4f, mape = %.4f, wmape = %.4f, loss = %.4f, dataset len = %d' % (epoch, test_mae, test_mape, test_wmape, test_loss, test_dataset_len))
    model.train().cuda()
    return test_loss, test_mae, test_mape, test_wmape

from torch.optim.lr_scheduler import _LRScheduler
class WarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, warm_up_iter, init_lr, max_lr, last_epoch=-1):
        self.m_iter = warm_up_iter
        self.max_lr = max_lr
        self.init_lr = init_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch <= self.m_iter:
                lr = base_lr + (self.last_epoch * 1.0) / self.m_iter * (self.max_lr - self.init_lr)
            else:
                lr = self.max_lr * (0.95 ** (self.last_epoch - 5))
            lrs.append(lr)
        return lrs

cate_shape = 78
cate_size = 5
seq_dim = 48 # seq_nume_size
flat_dim = 44 # flat_nume_size
cate_hidden = 5
hidden_dim = 64 # seq_hidden
extra_flat_dim = 274
static_size=24
mha_hidden = 32
n_head = 4

batch_size = 128
epochs = 20
verbose = True
target_name = ['battery_cost_4h']
model_version = '_v' + args.ver + '_'
print(model_version)
label_name = ['order_cnt_per_bike', 'battery_cost_per_order', 'distance_per_bike', 'battery_cost_per_bike']
print(label_name[int(args.label_index)])

for _label in range(len(target_name)):
    print('='*50)
    print(target_name[_label]+' test begin!')
    print('='*50)
    
    print('LR 1e-4 : 4e-3 | batch_size 256 |init| weighted | poisson | t-5 | v0')
    
    model = GruModel(cate_shape, cate_size, seq_dim, flat_dim, cate_hidden, hidden_dim, static_size, mha_hidden, n_head, extra_flat_dim, pre_train=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
#     warm_up_scheduler = WarmUpScheduler(optimizer, 5, 1e-4, 4e-3)
    
    model = model.train().cuda()

    best_loss = 99999999
    best_mae = 9999999
    best_wmape = 9999999
    best_loss_epoch = 0
    best_mae_epoch = 0
    best_wmape_epoch = 0
    for epoch in range(epochs):
        start_time = time.time()

#         learn_rate = warm_up_scheduler.get_lr()[0]
#         print("Epoch:%s, Learn_rate:%s" % (epoch, learn_rate))

        train_pred = []
        train_label = []
        train_loss = 0.
        train_cnt = 0

        for i in range(train_len):
            try:
                train_data = SegData(args.train_dir + "/part-%s" % i, 12, int(args.label_index))
            except:
                continue
            if len(train_data) == 0:
                continue
            myloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            for idx, batch in enumerate(myloader):
                geo, version, features, raw_label = batch
                if args.label_index == 2:
                    raw_label /= 1000
                flat_seq_c, flat_seq_d, flat_d, hh_seq_c, hh_seq_d, hh_seq_label, seq_c, seq_d, seq_label = get_feature(f = features, use_gpu = 1, feature_types = 9)

                y = raw_label.reshape(-1, 1).float().cuda()
                pred = model(flat_seq_c, flat_seq_d, flat_d,  hh_seq_c, hh_seq_d, hh_seq_label, seq_c[:, :, -1, :], seq_d[:, :, -1, :], seq_label[:, :, -1, :])
                if args.loss == "poisson" or args.loss == "mae":
                    loss = (loss_func(pred, y)).mean()
                elif args.loss == "tweedie":
                    loss = (loss_func(torch.exp(pred), y)).mean()
                
                train_loss += loss.data * len(y)
                train_cnt += len(y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_pred += pred.cpu().data.numpy().reshape(-1).tolist()
                train_label += y.cpu().data.numpy().reshape(-1).tolist()

        if verbose:
            train_pred = np.array(train_pred).reshape(-1)
            train_label = np.array(train_label).reshape(-1)
            train_loss /= train_cnt
            if args.loss == "poisson" or args.loss == "tweedie":
                train_mae = mean_absolute_error(train_label, decode_label(train_pred))
                train_mape = mape(train_label, decode_label(train_pred))
                train_wmape = wmape(train_label, decode_label(train_pred))
            elif args.loss == "mae":
                train_mae = mean_absolute_error(train_label, train_pred)
                train_mape = mape(train_label, train_pred)
                train_wmape = wmape(train_label, train_pred)
        
            print('Train predict std is %.4f' % (np.std(np.array(train_pred))))
            print('Train Iteration %d, mae = %.4f, mape = %.4f, wmape = %.4f, loss = %.4f' % (epoch, train_mae, train_mape, train_wmape, train_loss))
        
            test_loss, test_mae, test_mape, test_wmape = eval_result(batch_size, model, epoch)
            torch.save(model.state_dict(), args.model_dir + '/gru49_v2_' + target_name[int(_label)] + model_version + str(epoch) + '-th.model')
            print('spend {} second'.format(time.time() - start_time))
            print('')
        #warm_up_scheduler.step()
        if test_loss < best_loss:
            best_loss = test_loss
            best_loss_epoch = epoch
        if test_mae < best_mae:
            best_mae = test_mae
            best_mae_epoch = epoch
        if test_wmape < best_wmape:
            best_wmape = test_wmape
            best_wmape_epoch = epoch
    print('Best Loss = %.4f, Best Epoch = %d, Best_MAE = %.4f, Best Epoch = %d, Best_WMAPE = %.4f, Best Epoch = %d' % (best_loss, best_loss_epoch, best_mae, best_mae_epoch, best_wmape, best_wmape_epoch))