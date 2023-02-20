# -*-Encoding: utf-8 -*-
"""
Description: A CnnTst model to forecast future wind power
Date:    2022/07/16
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

#from tsai.all import TSTPlus
from tsai.models.TSTPlus import TSTPlus
#from TSTPlus_model import TSTPlus


class SeqCNN(nn.Module):
    def __init__(self, settings, in_dim=None, fc_output=False):
        super(SeqCNN, self).__init__()

        self.input_len = settings["input_len"]
        self.output_len = settings["output_len"]
        # self.out = settings["var_out"]
        # self.out_dim = settings["out_dim"]
        self.conv_layers = settings["conv_layers"]
        self.dropout = nn.Dropout(settings["fc_dropout"])
        self.fc_output = fc_output

        if in_dim is None:
            self.in_dim = settings["var_len"]
        else:
            self.in_dim = in_dim
        print("self.in_dim:", self.in_dim)
        self.cnn_out_dim = self.in_dim

        conv_kernel_size = 4
        conv_stride = 1
        conv_padding = int(np.floor(conv_kernel_size / (2 * conv_stride)))  # same padding
        dilation = 1
        print("conv_padding:", conv_padding)

        pool_kernel_size = 4
        pool_stride = 2
        pool_padding = int(np.floor(pool_kernel_size / (2 * pool_stride)))  # same padding
        print("pool_padding:", pool_padding)

        min_con_layers = int(np.ceil(math.log(settings["input_len"] / settings["output_len"], pool_stride)))
        if self.conv_layers < min_con_layers:
            self.conv_layers = int(np.ceil(math.log(settings["input_len"] / settings["output_len"], pool_stride)))

        print("self.conv_layers:", self.conv_layers)

        pool1d = nn.MaxPool1d(kernel_size=pool_kernel_size,
                     stride=pool_stride,
                     padding=pool_padding)

        if settings["pool_mod"] is 'avg':
            pool1d = nn.AvgPool1d(kernel_size=pool_kernel_size,
                                    stride=pool_stride,
                                     padding=pool_padding)
        conv_block = [
                    nn.Conv1d(in_channels=self.in_dim,
                              out_channels=self.cnn_out_dim,
                              kernel_size=conv_kernel_size,
                              stride=conv_stride,
                              padding=conv_padding,
                              padding_mode='replicate',
                              dilation=dilation),
                    nn.BatchNorm1d(num_features=self.cnn_out_dim),
                    nn.ReLU(),
                    # nn.GLU(),
                    pool1d,
                ]

        conv_module_list = nn.ModuleList()

        for _ in range(self.conv_layers):
            conv_module_list.extend(conv_block)

        self.seq_cnn = nn.Sequential(* conv_module_list)

        """
        # output length:
        cov_out_len = int(
            np.floor((self.input_len + 2 * conv_padding - dilation * (conv_kernel_size - 1) - 1) / conv_stride + 1))

        pool_out_len = int(
            np.floor((self.input_len + 2 * pool_padding - (pool_kernel_size - 1) - 1) / pool_stride + 1))
        """
        self.out_len = int(self.input_len/(pool_stride ** self.conv_layers))
        print("self.out_len:", self.out_len)
        self.out_dim = int(np.ceil(self.output_len / self.out_len))
        print("self.out_dim:", self.out_dim)
        self.fc_projection = nn.Linear(in_features=self.in_dim, out_features=self.out_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, embed_x):

        # print('embed_x size:', embed_x.shape)
        # batch_size x seq_len x embedding_size  -> batch_size x embedding_size x seq_len
        out = embed_x.permute(0, 2, 1)
        # print("out.shape: ", out.shape)

        out = self.seq_cnn(out)
        # print("out.shape1: ", out.shape)  # out[i]:batch_size x feature_size*out_dim
        out = out.permute(0, 2, 1)
        # print("out.shape2: ", out.shape)

        if self.fc_output:
            cut_len = int(self.output_len / self.out_dim)
            # print("cut_len: ", cut_len)
            out = self.fc_projection(self.dropout(out))
            # print("fc_projection out: ", out.shape)
            out = F.relu(
                out[:, -cut_len:, -self.out_dim:].reshape(-1, self.output_len, 1))  # [B, L, C]
            # print("final out: ", out.shape)
            return out
        else:
            return out


class CnnTst(nn.Module):
    """
    Desc:
        A CNN + TST model
    """

    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(CnnTst, self).__init__()
        self.output_len = settings["output_len"]
        self.var_len = settings["var_len"]
        self.nodes = settings["capacity"]
        # self.out = settings["var_out"]
        # self.out_dim = settings["out_dim"]
        # self.dropout = nn.Dropout(settings["fc_dropout"])
        # self.attention_mod = settings["attention_mod"]  # 'seq2v', 'self', 'both'
        # self.projection_mod = settings["projection_mod"]  # ['linear', 'ff']
        # self.hidden_dim = settings["hidden_dims"]
        self.cat_var = 3

        self.id_emb_dim = 8
        self.t_emb_dim = 8
        self.w_emb_dim = 4

        self.id_emb = nn.Embedding(self.nodes + 1, self.id_emb_dim)
        self.t_emb = nn.Embedding(settings["day_len"], self.t_emb_dim)
        self.w_emb = nn.Embedding(7, self.w_emb_dim)

        self.in_dim = (self.var_len - self.cat_var + self.id_emb_dim )
        # self.in_dim = (self.var_len - self.cat_var + self.id_emb_dim + self.t_emb_dim + self.w_emb_dim)
        print('CnnTst self.in_dim:', self.in_dim)

        self.cnn_model = SeqCNN(settings, in_dim=self.in_dim)

        self.tst_model = TSTPlus(c_in=self.in_dim,
                                 c_out=settings["output_len"],
                                 seq_len=self.cnn_model.out_len,
                                max_seq_len=settings["max_seq_len"],
                                n_layers=settings["n_layers"],
                                n_heads=settings["n_heads"],
                                d_model=settings["d_model"],
                                d_ff=settings["d_ff"],
                                attn_dropout=settings["attn_dropout"],
                                dropout=settings["dropout"],
                                fc_dropout=settings["fc_dropout"])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x_enc):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:(B, T_in, F_in)
        Returns:
            A tensor: (B, T_out, , F_out)
        """
        #print('x_enc.shape:', x_enc.shape)

        turb_id = self.id_emb(x_enc[..., 0].type(torch.int32))
        # weekday_id = self.w_emb(x_enc[..., 1].type(torch.int32))
        # time_id = self.t_emb(x_enc[..., 2].type(torch.int32))
        #x_enc = torch.concat((x_enc[..., self.cat_var:], turb_id, weekday_id, time_id), -1)
        x_enc = torch.concat((x_enc[..., self.cat_var:], turb_id), -1)
        # print('x_enc:', x_enc.size())

        out = self.cnn_model(x_enc)
        # print('out 1:', out.size())

        out = out.permute(0, 2, 1)
        out = self.tst_model(out)

        # print('out 2:', out.size())

        return out

