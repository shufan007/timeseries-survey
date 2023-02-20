import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def encode_label(x):
    return np.log(x + 1)
def decode_label(x):
    return np.exp(x)


class SegData(Dataset):
    def __init__(self, file_name, input_len, label_index=-1, current=False, day=False):
        self.samples = torch.load(file_name)
        self.input_len = input_len
        self.label_index = label_index
        self.current = bool(current)
        self.day = bool(day)

    def __getitem__(self, idx):
        """
        :self.samples idx:
                0: geo
                1: version
                2: label
                3: current weather, length:5
                4: current continuous features, length:44
                5: current statistical features, length:274
                6: historic 17 hours weather, length:5
                7: historic 17 hours continuous features, length:44
                8: historic 17 hours labels, length:4
                9: historic 49 day weather, length:5
                10: historic 49 day continuous features, length:44
                11: historic 49 day labels, length:4
        """
        # geo = self.samples[0][idx]
        raw_label = self.samples[2][idx][self.label_index]

        h17tmp = []
        _h17tmp = self.samples[7][idx]
        for i in range(44):
            if i not in [0, 2, 4, 11, 12, 17, 23]:  # 44 - 7 = 37
                h17tmp.append(_h17tmp[..., i])
        h17tmp = torch.stack(h17tmp, dim=-1)
        h17_seq_features = torch.cat([self.samples[6][idx], h17tmp, self.samples[8][idx]], axis=-1)   # 37+4+5 = 46
        # raw_label shaped [bs, 1],
        # d49_seq_features = torch.cat([self.samples[9][idx], self.samples[10][idx], self.samples[11][idx]], axis = -1)

        # cur_features shaped [bs, 49],  here 49 = 5 + 44
        # h17_seq_features shaped [bs, 17, 53], d49_seq_features shaped [bs, 49, 5, 53] , here 53 = 5 + 44 + 4
        cur_features, day_features = raw_label, raw_label
        if self.current == True:
            cur_features = []
            for i in range(44):
                if i not in [0, 2, 4, 11, 12, 17, 23]:  # 44 - 7 = 37
                    cur_features.append(self.samples[4][idx][..., i])
            cur_features = torch.stack(cur_features, dim=-1)
            cur_features = torch.cat([self.samples[3][idx], cur_features], axis=-1)
        if self.day == True:
            day_features = []
            _daytmp = self.samples[10][idx]
            for i in range(44):
                if i not in [0, 2, 4, 11, 12, 17, 23]:  # 44 - 7 = 37
                    day_features.append(_daytmp[..., i])
            day_features = torch.stack(day_features, dim=-1)
            day_features = torch.cat([self.samples[9][idx], day_features, self.samples[11][idx]], axis=-1)
        return cur_features, h17_seq_features, day_features, raw_label

    def __len__(self):
        return self.samples[1].shape[0]


class SegData_orignal(Dataset):
    def __init__(self, file_name, input_len, label_index=-1):
        self.samples = torch.load(file_name)
        self.input_len = input_len
        self.label_index = label_index

    def __getitem__(self, idx):
        raw_label = self.samples[2][idx][self.label_index]  # ['order_cnt_per_bike', 'battery_cost_per_order', 'distance_per_bike', 'battery_cost_per_bike']
        features = []
        for i in range(3, self.input_len):
            features.append(self.samples[i][idx])
        return features, raw_label

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

