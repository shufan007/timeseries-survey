import dgl
import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.sparse.linalg import eigs

def time2obj(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    return data_sj

def time2int(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    time_int = int(time.mktime(data_sj))
    return time_int

def int2time(t):
    timestamp = datetime.datetime.fromtimestamp(t)
    return timestamp.strftime('"%H:%M"')

def func_add_t(x):
    time_strip = 600
    time_obj = time2obj(x)
    time_e = ((
        (time_obj.tm_sec + time_obj.tm_min * 60 + time_obj.tm_hour * 3600)) //
              time_strip) % 288
    return time_e

def func_add_h(x):
    time_obj = time2obj(x)
    hour_e = time_obj.tm_hour
    return hour_e

class PGL4WPFDataset(Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    15 days for training,
                        3 days for validation,
                        and 6 days for testing
    """

    def __init__(
            self,
            data_path,
            filename='wtb5_10.csv',
            flag='train',
            size=None,
            capacity=134,
            day_len=144,
            day_seq=15,
            train_days=214,  # 15 days
            val_days=31,  # 3 days
            test_days=0,  # 6 days
            total_days=245,  # 30 days
            theta=0.9, ):

        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]

        self.start_col = 0
        self.capacity = capacity
        self.theta = theta
        self.day_seq = day_seq

        # initialization
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.data_path = data_path
        self.filename = filename

        self.total_size = self.unit_size * total_days
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size

        self.test_size = test_days * self.unit_size
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        df_data, raw_df_data = self.data_preprocess(df_raw)
        del df_raw
        data_x, graph = self.build_graph_data(df_data, raw_df_data)
        del df_data, raw_df_data
        print(f"***************{self.flag}**************data_shape: {data_x.shape}")
        self.data_x = data_x
        # self.data2_x = self.build_day_seq_data(self.data_x)
        del data_x
        self.graph = graph

    def __getitem__(self, index):
        # Sliding window with the size of input_len + output_len
        d_begin = index
        d_end = d_begin + self.day_seq * 144
        s_begin = d_end
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        day_x = self.build_day_seq(self.data_x[:, d_begin:d_end, 2:])
        seq_x = self.data_x[:, s_begin:s_end, :]
        seq_y = self.data_x[:, r_begin:r_end, :]

        if self.flag == "train":
            perm = np.arange(0, seq_x.shape[0])
            np.random.shuffle(perm)
            return day_x[perm], seq_x[perm], seq_y[perm]
        else:
            return day_x, seq_x, seq_y

    def build_day_seq(self, day_data, day):
        sbegin = 0
        new = []
        for i in range(day):
            send = sbegin + 144
            new.append(day_data[:, sbegin:send, :])
            sbegin = 144 * (i + 1)
        return np.sum(np.stack(new, axis=1), axis=2)

    def __len__(self):
        return self.data_x.shape[1] - self.input_len - self.output_len + 1

    def data_preprocess(self, df_data):

        feature_name = [
            n for n in df_data.columns
            if "Patv" not in n and 'Day' not in n and 'Tmstamp' not in n and
            'TurbID' not in n
        ]
        feature_name.append("Patv")

        new_df_data = df_data[feature_name]

        print(self.flag, '____adding time')
        t = df_data['Tmstamp'].apply(func_add_t)
        new_df_data.insert(0, 'time', t)

        weekday = df_data['Day'].apply(lambda x: x % 7)
        new_df_data.insert(0, 'weekday', weekday)

        pd.set_option('mode.chained_assignment', None)

        return new_df_data.replace(to_replace=np.nan, value=0, inplace=False), new_df_data

    def get_raw_df(self):
        return self.raw_df

    def build_graph_data(self, data, raw_df_data):
        cols_data = data.columns[self.start_col:]
        data = data[cols_data]
        raw_df_data = raw_df_data[cols_data]
        data = data.values
        data = np.reshape(data, [self.capacity, self.total_size, len(cols_data)])
        raw_df_data = raw_df_data.values
        raw_df_data = np.reshape( raw_df_data, [self.capacity, self.total_size, len(cols_data)])

        border1s = [
            0, self.train_size - self.input_len,
            self.train_size + self.val_size - self.input_len
        ]
        border2s = [
            self.train_size, self.train_size + self.val_size,
            self.train_size + self.val_size + self.test_size
        ]
        if self.flag == 'train':
            self.data_mean = np.expand_dims(
                np.mean(data[:, border1s[0]:border2s[0], 2:], axis=1, keepdims=True),
                0)
            self.data_scale = np.expand_dims(
                np.std(data[:, border1s[0]:border2s[0], 2:],axis=1, keepdims=True),
                0)

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.raw_df = []
        for turb_id in range(self.capacity):
            self.raw_df.append(
                pd.DataFrame(
                    data=raw_df_data[turb_id, border1 + self.input_len:border2],
                    columns=cols_data))
        del raw_df_data
        data_edge = data[:, border1s[0]:border2s[0], -1]
        edge_w = np.corrcoef(data_edge)

        k = 5
        topk_indices = np.argpartition(edge_w, -k, axis=1)[:, -k:]
        rows, _ = np.indices((edge_w.shape[0], k))
        kth_vals = edge_w[rows, topk_indices].min(axis=1).reshape([-1, 1])

        row, col = np.where(edge_w > kth_vals)
        graph = dgl.graph((row, col), num_nodes=self.capacity)
        # graph.add_nodes(self.capacity)
        # graph.add_edges(row, col)
        graph.add_edges(col, row)
        A = np.zeros((int(self.capacity), int(self.capacity)))
        l = len(row)
        for i in range(l):
            r, c = int(row[i]), int(col[i])
            A[r, c] = 1
            A[c, r] = 1
        A = self.scaled_Laplacian(A)
        return data[:, border1:border2, :], A
    def scaled_Laplacian(self, W):
        assert W.shape[0] == W.shape[1]
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        lambda_max = eigs(L, k=1, which='LR')[0].real
        return (2 * L) / lambda_max - np.identity(W.shape[0])
if __name__ == "__main__":
    data_path = "./data"
    data = PGL4WPFDataset(data_path, filename="wtbdata_245days.csv")
    adj = data.graph
    data_x = data.data_x
    # data2_x = data.data2_x
    print(adj.shape, data_x)
    # import networkx as nx
    #
    # # Since the actual graph is undirected, we convert it for visualization
    # nx_G = adj.to_networkx().to_undirected()
    # # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    # pos = nx.kamada_kawai_layout(nx_G)
    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # print(nx_G)
    # plt.show()
