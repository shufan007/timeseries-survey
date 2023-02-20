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
        #self.out = settings["var_out"]
        self.out = settings["out_dim"]
        self.nhead = settings["nhead"]
        self.dropout = nn.Dropout(settings["dropout_proj"])
        self.attention_mod = settings["attention_mod"] # 'seq2v', 'self', 'both'
        self.projection_mod = settings["projection_mod"] # ['linear', 'ff'] 
        
        self.hidden_dim = settings["hidden_dims"]
        self.cat_var = 2

        t_emb_dim = 8
        w_emb_dim = 4
        cat_dim = t_emb_dim + w_emb_dim

        self.t_emb = nn.Embedding(144, t_emb_dim)
        self.w_emb = nn.Embedding(7, w_emb_dim)

        #self.enc_dim = (self.in_dim - self.cat_var + self.t_emb_dim)
        self.enc_dim = self.in_dim
        
        #self.enc_dim = (self.in_dim - self.cat_var )
        #onehead_embed_dim = self.enc_dim * self.nodes
        #embed_dim = onehead_embed_dim * self.nhead
        
        embed_dim = self.enc_dim * self.nodes
        #num_head = self.nodes
        num_head = self.enc_dim
        
        vdim = 134

        self.att_seq2cat = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_head,
                                         dropout=settings["dropout_att"],
                                         batch_first=True,
                                         kdim=cat_dim,
                                         vdim=embed_dim)

        self.att_seq2v = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_head,
                                         dropout=settings["dropout_att"],
                                         batch_first=True,
                                         kdim=vdim,
                                         vdim=embed_dim)
        
        self.att2self = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_head,
                                         dropout=settings["dropout_att"],
                                         batch_first=True,
                                         kdim=embed_dim,
                                         vdim=embed_dim)
        
        self.gru = nn.GRU(input_size=self.enc_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=settings["num_layers"],
                          batch_first=True)
        
        self.gru2 = nn.GRU(input_size=self.enc_dim * 2,
                          hidden_size=self.hidden_dim,
                          num_layers=settings["num_layers"],
                          batch_first=True)
        
        #self.glu = GLU(self.hidden_dim)
        #self.projection_last = nn.Linear(self.hidden_dim, self.output_len) 
        self.projection = nn.Linear(self.hidden_dim, self.out)
        
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
            , nn.Linear(down_dim, self.out)
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
        batch_size, seq_len, invar = x_enc.size(0), x_enc.size(2), x_enc.size(3)
        
        """
        time_idx = 1
        x_time_id = x_enc[..., time_idx]
        x_time_id = x_time_id.type(torch.int32)   # [b, t, 2]
        time_id = self.t_emb(x_time_id)
        #weekday_id = self.w_emb(x_time_id[..., 0])
        x_enc = torch.concat((x_enc, time_id), -1)  
        """
        x_enc = x_enc.permute(0, 2, 1, 3)
        v_enc = x_enc[..., -1]
        x_enc = x_enc.reshape(batch_size, seq_len, self.nodes * self.enc_dim)
        
        if self.attention_mod == 'both':
            x_att_seq2v, _ = self.att_seq2v(x_enc, v_enc, x_enc)
            x_att_self, _ = self.att2self(x_enc, x_enc, x_enc)
            x_att = torch.concat((x_att_seq2v, x_att_self), -1)  
            
            x_gru_in = x_att.reshape(batch_size, seq_len, self.nodes, self.enc_dim * 2)
            x_gru_in = x_gru_in.permute(0, 2, 1, 3).reshape(-1, seq_len, self.enc_dim * 2)
            del x_att, x_enc
            
            dec, _ = self.gru2(x_gru_in)
            del x_gru_in
            
        else:
            if self.attention_mod == 'seq2v':
                x_att, _ = self.att_seq2v(x_enc, v_enc, x_enc)
            else:
                x_att, _ = self.att2self(x_enc, x_enc, x_enc)
        
            #print(x_enc.shape)  # [bs, seq_len, d_model]
            x_gru_in = x_att.reshape(batch_size, seq_len, self.nodes, self.enc_dim)
            x_gru_in = x_gru_in.permute(0, 2, 1, 3).reshape(-1, seq_len, self.enc_dim)
            del x_att, x_enc
            
            dec, _ = self.gru(x_gru_in)
            # print('******* ', torch.cuda.memory_allocated() / 1024 / 1024)
            del x_gru_in
            
        # print('******* ', torch.cuda.memory_allocated() / 1024 / 1024)
        #print('dec: ', dec.shape)
        #dec = self.glu(dec)
        cut_len = int(self.output_len/self.out)
        if self.projection_mod == 'ff':
            dec = self.projection_v2(dec)
            dec = dec[:, -cut_len:, -self.out:].reshape(batch_size, self.nodes, self.output_len)  
        else:  # self.projection_mod == 'linear':
            dec = self.projection(self.dropout(dec))
            dec = F.relu(dec[:, -cut_len:, -self.out:].reshape(batch_size, self.nodes, self.output_len)) # [B, N, L, D]
        
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