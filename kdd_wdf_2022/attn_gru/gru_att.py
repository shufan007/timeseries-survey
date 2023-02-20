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
import torch.nn.functional as F
import numpy as np

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
        self.input_len = settings["input_len"]
        self.output_len = settings["output_len"]
        self.var_len = settings["var_len"]
        self.nodes = settings["capacity"]
        #self.out = settings["var_out"]
        # self.out_dim = settings["out_dim"]
        self.nhead = settings["nhead"]
        self.dropout = nn.Dropout(settings["dropout_proj"])
        self.is_attention = settings["is_attention"]
        self.attention_mod = settings["attention_mod"] # 'seq2cat', 'self'
        self.projection_mod = settings["projection_mod"] # ['linear', 'ff']      
        self.hidden_dim = settings["hidden_dims"]
        self.cat_var = settings["cat_var_len"]  # 3

        self.id_emb_dim = 8
        self.t_emb_dim = 8
        self.w_emb_dim = 4
        #kdim = t_emb_dim
        cat_dim = self.t_emb_dim + self.w_emb_dim

        self.id_emb = nn.Embedding(self.nodes + 1, self.id_emb_dim)
        self.t_emb = nn.Embedding(settings["day_len"], self.t_emb_dim)
        self.w_emb = nn.Embedding(7, self.w_emb_dim)

        self.attn_in_dim = (self.id_emb_dim + self.var_len - 1)

        num_heads = self.attn_in_dim
        embed_dim = num_heads * self.nodes
        
        self.att_seq2cat = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=settings["dropout_att"],
                                         batch_first=True,
                                         kdim=cat_dim,
                                         vdim=embed_dim)

        self.att_self = nn.MultiheadAttention(embed_dim=embed_dim,
                                              num_heads=num_heads,
                                              dropout=settings["dropout_att"],
                                              batch_first=True,
                                              kdim=embed_dim,
                                              vdim=embed_dim)

        self.attn = self.att_self
        if self.attention_mod is 'seq2cat':
            self.attn = self.att_seq2cat

        if self.is_attention:
            self.gru_in_dim = 2 * self.attn_in_dim  # concat attn var
        else:
            self.gru_in_dim = self.attn_in_dim

        self.gru = nn.GRU(input_size=self.gru_in_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=settings["num_layers"],
                          batch_first=True)
        #self.glu = GLU(self.hidden_dim)

        self.out_dim = int(np.ceil(self.output_len / self.input_len))

        #self.projection_last = nn.Linear(self.hidden_dim, self.output_len) 
        self.projection = nn.Linear(self.hidden_dim, self.out_dim)
        
        dim_decay = 2
        down_dim = int(self.hidden_dim/dim_decay)
        self.projection_ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim)
            #, nn.BatchNorm1d(self.hidden_dim)
            , nn.Tanh()
            , nn.Linear(self.hidden_dim, self.hidden_dim)
            #, nn.BatchNorm1d(self.hidden_dim)
            , nn.Tanh()
            , nn.Linear(self.hidden_dim, down_dim)
            #, nn.BatchNorm1d(down_dim)
            , nn.Tanh()
            , nn.Linear(down_dim, down_dim)
            #, nn.BatchNorm1d(down_dim)
            , nn.Tanh()
            , nn.Dropout(p=settings["dropout_proj"])
            , nn.Linear(down_dim, self.out_dim)
        )
        
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
            x_enc:(B, N_nodes, T_in, F_in)
        Returns:
            A tensor: (B, N_nodes, T_out, , F_out)
        """
        #print(x_enc.shape)
        batch_size,  seq_len, invar = x_enc.size(0),  x_enc.size(2), x_enc.size(3)

        if self.attention_mod is 'seq2cat':
            weekday_id = self.w_emb(x_enc[:, 0, :, 1].type(torch.int32))
            time_id = self.t_emb(x_enc[:, 0, :, 2].type(torch.int32))
            x_cat_emb = torch.concat((time_id, weekday_id), -1)
            #print("x_cat_emb.shape: ", x_cat_emb.shape)

        turb_id = self.id_emb(x_enc[..., 0].type(torch.int32))
        #print("x_enc.shape 1: ", x_enc.shape)
        x_enc = torch.concat((turb_id, x_enc[..., 1:]), -1)
        #print("x_enc.shape 2: ", x_enc.shape)
        x_attn_in = x_enc.permute(0, 2, 1, 3)
        #print("x_enc.shape 3: ", x_enc.shape)

        # attn_invar = invar - 1 + self.id_emb_dim   # self.attn_in_dim
        #print("invar 2: ", invar)
        #x_att, _ = self.att_seq2cat(x_enc.reshape(batch_size, seq_len, self.nodes * invar), x_cat, x_enc[..., -1])
        x_gru_in = x_enc

        if self.is_attention:
            Q = K = V = x_attn_in.reshape(batch_size, seq_len, self.nodes * self.attn_in_dim)
            if self.attention_mod is 'seq2cat':
                K = x_cat_emb

            x_att, _ = self.attn(Q, K, V)
            #print(x_enc.shape)  # [bs, seq_len, d_model]
            #x_gru_in = x_enc * x_att.reshape(batch_size, seq_len, self.nodes, invar)
            x_att = x_att.reshape(batch_size, seq_len, self.nodes, self.attn_in_dim)

            x_gru_in = torch.cat([x_gru_in, x_att.permute(0, 2, 1, 3)], axis=-1)

        print("x_gru_in.shape1: ", x_gru_in.shape)
        print("self.attn_in_dim:", self.attn_in_dim)
        print("self.gru_in_dim:", self.gru_in_dim)
        #gru_invar = invar - 1 + attn_invar
        x_gru_in = x_gru_in.reshape(-1, seq_len, self.gru_in_dim)

        print("x_gru_in.shape2: ", x_gru_in.shape)
        #del x_att, x_enc, x_time_id
        dec, _ = self.gru(x_gru_in)
        #del x_gru_in
        # print('******* ', torch.cuda.memory_allocated() / 1024 / 1024)
        #dec = self.glu(dec)

        print("dec.shape1: ", dec.shape)

        cut_len = int(self.output_len/self.out_dim)
        if self.projection_mod == 'ff':
            dec = self.projection_v2(dec)
            dec = dec[:, -cut_len:, -self.out_dim:].reshape(batch_size, self.nodes, self.output_len)
        else:  # self.projection_mod == 'linear':
            dec = self.projection(self.dropout(dec))
            # print("dec.shape2: ", dec.shape)
            dec = F.relu(dec[:, -cut_len:, -self.out_dim:].reshape(batch_size, self.nodes, self.output_len)) # [B, N, L, D]
       
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