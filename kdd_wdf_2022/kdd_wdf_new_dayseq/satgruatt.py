
from layers import GLU
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cheb_polynomial(L_tilde, K):
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials

class SatGruAtt(nn.Module):

    def __init__(self, settings):
        # type: (dict) -> None
        super(SatGruAtt, self).__init__()
        self.output_len = settings["output_len"]
        self.seq_len = settings["input_len"]
        self.in_dim = settings["var_len"]
        self.nodes = settings["capacity"]
        self.out = settings["var_out"]
        self.nhead = settings["nhead"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.hidden_dim = settings["hidden_dims"]
        self.day_seq_len = settings["day_seq_len"]
        self.init_len = settings["day_len"]
        self.cat_var = 2

        # encoder
        self.dec_emb = nn.Linear(self.in_dim - 2, 2 * self.hidden_dim)
        self.conv_block = nn.ModuleList([
            nn.Conv1d(2 * self.hidden_dim, self.hidden_dim, kernel_size=36, dilation=1),
            nn.Conv2d(2 * self.hidden_dim, self.hidden_dim, kernel_size=(6, 3), dilation=(1, 1), padding=(1, 1)),
            nn.Conv1d(2 * self.hidden_dim, self.hidden_dim, kernel_size=36, dilation=2),
        ])

        self.seq_att = nn.MultiheadAttention(embed_dim=self.hidden_dim,
                                         num_heads=self.nhead,
                                         dropout=settings["dropout"],
                                         batch_first=True)
        self.seq_gru = nn.GRU(input_size=self.hidden_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=settings["num_layers"],
                          batch_first=True)
        self.projection = nn.Linear(self.hidden_dim, self.out)
        # self.projection = nn.Conv2d(self.seq_len + self.day_seq_len, self.output_len, kernel_size=(1, self.hidden_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    def forward(self, day_seq, adj):
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:(B, N_nodes, T_in, F_in)
        Returns:
            A tensor: (B, N_nodes, T_out, , F_out)
        """
        bs, nodes, _, _, _ = day_seq.shape
        day_seq = self.dec_emb(day_seq.reshape(-1, self.day_seq_len, self.init_len, self.in_dim - 2)).permute(0, 3,1,2)  # b, f, d, t

        day_seq1 = self.conv_block[0](day_seq.reshape(bs * nodes, self.hidden_dim * 2, -1))  # [B, T_day, D]
        day_seq2 = self.conv_block[1](day_seq).reshape(bs * nodes, self.hidden_dim, -1)
        day_seq3 = self.conv_block[2](day_seq.reshape(bs * nodes, self.hidden_dim * 2, -1))
        # print(day_seq1.shape, day_seq2.shape, day_seq3.shape, day_seq4.shape)
        day_seq = day_seq1[..., -self.output_len * 2:] + day_seq2[..., -self.output_len * 2:] + day_seq3[...,
                                                                                            -self.output_len * 2:]
        day_seq = day_seq.permute(0, 2, 1)
        del day_seq1, day_seq2, day_seq3, adj

        x_att, _ = self.seq_att(day_seq, day_seq, day_seq)
        x_gru_in = day_seq * x_att
        del x_att
        seq_dec, _ = self.seq_gru(x_gru_in)
        del x_gru_in
        seq_dec = seq_dec[:, -self.output_len:,:]
        # dec = self.glu(dec.permute(0, 2, 1, 3))
        dec = self.projection(seq_dec).reshape(bs, self.nodes, -1)
        del seq_dec
        return dec


