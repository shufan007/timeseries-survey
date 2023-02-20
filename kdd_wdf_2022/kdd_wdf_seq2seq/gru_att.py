# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: A GRU-based baseline model to forecast future wind power
Authors: Lu,Xinjiang (luxinjiang@baidu.com), Li,Yan (liyan77@baidu.com)
Date:    2022/03/10
"""
import torch
import torch.nn as nn
from layers import GLU
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GruAtt(nn.Module):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(GruAtt, self).__init__()
        self.output_len = settings["output_len"]
        self.in_dim = settings["var_len"]
        self.nodes = settings["capacity"]
        self.out = settings["var_out"]
        self.nhead = settings["nhead"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.hidden_dim = 64
        self.cat_var = 2
        self.t_emb = nn.Embedding(144, 143)
        self.w_emb = nn.Embedding(7, 6)
        self.att = nn.MultiheadAttention(embed_dim=(self.in_dim - self.cat_var) * self.nodes,
                                         num_heads=(self.in_dim - self.cat_var),
                                         dropout=settings["dropout"],
                                         batch_first=True,
                                         kdim=149,
                                         vdim=134)
        self.gru = nn.GRU(input_size=(self.in_dim - self.cat_var),
                          hidden_size=self.hidden_dim,
                          num_layers=settings["num_layers"],
                          batch_first=True)
        #self.glu = GLU(self.hidden_dim)
        self.projection = nn.Linear(self.hidden_dim, self.out)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    def forward(self, x_enc, x_time_id):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:(B, N_nodes, T_in, F_in)
        Returns:
            A tensor: (B, N_nodes, T_out, , F_out)
        """
        #print(x_enc.shape)
        batch_size,  seq_len, invar = x_enc.size(0),  x_enc.size(2), x_enc.size(3)
        x_time_id = x_time_id.type(torch.int32)   # [b, t, 2]
        time_id = self.t_emb(x_time_id[..., 1])
        weekday_id = self.w_emb(x_time_id[..., 0])
        x_enc = x_enc.permute(0, 2, 1, 3)

        x_cat = torch.concat((time_id, weekday_id), -1)

        x_att, _ = self.att(x_enc.reshape(batch_size, seq_len, self.nodes * invar), x_cat, x_enc[..., -1])
        #print(x_enc.shape)  # [bs, seq_len, d_model]
        x_gru_in = x_enc * x_att.reshape(batch_size, seq_len, self.nodes, invar)
        x_gru_in = x_gru_in.permute(0, 2, 1, 3).reshape(-1, seq_len, invar)
        #print('x_enc: ', x_enc.shape)
        # print('******* ', torch.cuda.memory_allocated() / 1024 / 1024)
        del x_att, x_enc, x_time_id
        # print('******* ', torch.cuda.memory_allocated() / 1024 / 1024)
        dec, _ = self.gru(x_gru_in)
        # print('******* ', torch.cuda.memory_allocated() / 1024 / 1024)
        del x_gru_in
        # print('******* ', torch.cuda.memory_allocated() / 1024 / 1024)
        #print('dec: ', dec.shape)
        #dec = self.glu(dec)
        dec = self.projection(self.dropout(dec))
        #print('dec: ', dec.shape)
        dec = F.relu(dec[:, -self.output_len:, -self.out:].reshape(batch_size, self.nodes, self.output_len)) # [B, N, L, D]
        return dec

class Spatial_Attention_layer(nn.Module):
    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices))

    def forward(self, x, x_time_id, y_time_id):
        '''
        :param x: (batch_size, N, F_in, T)
        :param x_time_id: (batch_size, T, 2)
        :param y_time_id: (batch_size, T, 2)
        :return: (B,N,N)
        '''
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        S_normalized = F.softmax(S, dim=1)
        return S_normalized