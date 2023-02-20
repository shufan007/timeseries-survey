
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

WIN = 3
DECOMP = 25
from dgl.nn.pytorch import GraphConv

class SeriesDecomp(nn.Module):
    """Ideas comes from AutoFormer
    Decompose a time series into trends and seasonal
    Refs:  https://arxiv.org/abs/2106.13008
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.padding = kernel_size/2 if torch.__version__ >= '1.5.0' else kernel_size/2 + 1
        self.kernel_size = kernel_size
        print(int(self.padding))

    def forward(self, x):
        t_x = x.permute(0, 2, 1)
        mean_x = F.avg_pool1d(
            t_x, self.kernel_size, stride=1, padding=int(self.padding))
        mean_x = mean_x.permute(0, 2, 1)
        return x - mean_x, mean_x


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder with Time Series Decomposition
    Ideas comes from AutoFormer
    Decoding trends and seasonal
    Decompose a time series into trends and seasonal
    Refs:  https://arxiv.org/abs/2106.13008
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dims_feedforward,
                 dropout=0.1,
                 activation="gelu",
                 attn_dropout=None,
                 act_dropout=None,
                 trends_out=134,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout

        self.decomp = SeriesDecomp(DECOMP)

        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        # self.cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dims_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dims_feedforward, d_model)
        self.padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.linear_trend = nn.Conv1d(d_model, trends_out, kernel_size=WIN, padding=self.padding)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, src, memory, src_mask=None, cache=None):
        residual = src
        src, _ = self.self_attn(src, src, src, None)
        src = residual + self.dropout1(src)
        src, trend1 = self.decomp(src)

        # residual = src
        # src, _ = self.self_attn(src, memory, memory, None)
        # src = residual + self.dropout1(src)
        #
        # src, trend2 = self.decomp(src)
        #    pass
        residual = src

        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)

        src, trend3 = self.decomp(src)
        res_trend = trend1 + trend3
        res_trend = self.linear_trend(res_trend.permute(0, 2, 1))
        return src, res_trend.permute(0, 2, 1)


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder with Time Series Decomposition
    Ideas comes from AutoFormer
    Decoding trends and seasonal
    Decompose a time series into trends and seasonal
    Refs:  https://arxiv.org/abs/2106.13008
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dims_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout

        self.decomp = SeriesDecomp(DECOMP)

        self.self_attn = nn.MultiheadAttention(d_model, nhead)

        self.linear1 = nn.Linear(d_model, dims_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dims_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None, cache=None):
        residual = src
        src, _ = self.self_attn(src, src, src, None)
        src = residual + self.dropout1(src)

        src, _ = self.decomp(src)

        residual = src

        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)

        src, _ = self.decomp(src)
        return src


class Encoder(nn.Module):
    def __init__(self, var_len, input_len, output_len, hidden_dims, nhead, encoder_layers, dropout):
        super(Encoder, self).__init__()
        self.var_len = var_len
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dims = hidden_dims
        self.nhead = nhead
        self.num_encoder_layer = encoder_layers

        self.enc_lins = nn.ModuleList()
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)
        for _ in range(self.num_encoder_layer):
            self.enc_lins.append(
                TransformerEncoderLayer(
                    d_model=self.hidden_dims,
                    nhead=self.nhead,
                    dropout=self.dropout,
                    activation="gelu",
                    attn_dropout=self.dropout,
                    act_dropout=self.dropout,
                    dims_feedforward=self.hidden_dims * 2))

    def forward(self, batch_x):
        for lin in self.enc_lins:
            batch_x = lin(batch_x)
        batch_x = self.drop(batch_x)
        return batch_x


class Decoder(nn.Module):
    def __init__(self, var_len, input_len, output_len, hidden_dims, nhead, decoder_layers, dropout, capacity):
        super(Decoder, self).__init__()
        self.var_len = var_len
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dims = hidden_dims
        self.nhead = nhead
        self.num_decoder_layer = decoder_layers

        self.dec_lins = nn.ModuleList()
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)
        self.capacity = capacity

        for _ in range(self.num_decoder_layer):
            self.dec_lins.append(
                TransformerDecoderLayer(
                    d_model=self.hidden_dims,
                    nhead=self.nhead,
                    dropout=self.dropout,
                    activation="gelu",
                    attn_dropout=self.dropout,
                    act_dropout=self.dropout,
                    dims_feedforward=self.hidden_dims * 2,
                    trends_out=self.capacity))

    def forward(self, season, trend, enc_output):
        for lin in self.dec_lins:
            season, trend_part = lin(season, enc_output)
            trend = trend + trend_part
        return season, trend

class Spatial_Attention_layer(nn.Module):
    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        S_normalized = F.softmax(S, dim=1)
        return S_normalized
class SpatialTemporalConv(nn.Module):
    """ Spatial Temporal Embedding
    Apply GAT and Conv1D based on Temporal and Spatial Correlation
    """

    def __init__(self, id_len, input_dim, output_dim, seq_len):
        super(SpatialTemporalConv, self).__init__()
        self.padding = 1 if torch.__version__ >= '1.5.0' else 2

        self.conv1 = nn.Conv1d(
            id_len * input_dim,
            output_dim,
            kernel_size=WIN,
            padding=self.padding,
            bias=False)
        self.id_len = id_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)
        self.spatio_att = Spatial_Attention_layer(input_dim, id_len, seq_len)
    # def _send_attention(self, src_feat, dst_feat, edge_feat):
    #     alpha = src_feat["k"] * dst_feat["q"]
    #     alpha = torch.sum(alpha, -1, keepdim=True)
    #     return {"alpha": alpha, "output_series": src_feat["v"]}
    #
    # def _reduce_attention(self, msg):
    #     alpha = msg.reduce_softmax(msg["alpha"])
    #     return msg.reduce(msg["output_series"] * alpha, pool_type="sum")

    def forward(self, x, adj):
        bz, seqlen, _ = x.shape
        x = x.reshape(bz, seqlen, self.id_len, self.input_dim)
        spa_att = self.spatio_att(x.permute(0, 2, 3, 1))    #b, n, n
        x = torch.matmul(x.permute(0, 1, 3, 2).reshape(bz, -1, self.id_len), spa_att)
        x = x.reshape(bz, seqlen, self.input_dim, self.id_len).permute(0, 2, 1, 3)    #b, f, t, n
        x = torch.einsum('bftn, nn -> bftn', x, adj)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(bz, seqlen, self.id_len * self.input_dim)
        x = self.conv1(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class WPFModel(nn.Module):
    def __init__(self, settings):
        super(WPFModel, self).__init__()
        self.var_len = settings["var_len"] - 2
        self.input_len = settings["input_len"]
        self.output_len = settings["output_len"]
        self.hidden_dims = settings["hidden_dims"]
        self.capacity = settings["capacity"]
        self.n_heads = settings["nhead"]
        self.num_layers = settings["num_layers"]
        self.dropout = settings["dropout"]

        self.decomp = SeriesDecomp(DECOMP)

        self.t_emb = nn.Embedding(300, self.hidden_dims)
        self.w_emb = nn.Embedding(300, self.hidden_dims)

        self.t_dec_emb = nn.Embedding(300, self.hidden_dims)
        self.w_dec_emb = nn.Embedding(300, self.hidden_dims)

        self.pos_dec_emb = nn.Parameter(torch.FloatTensor( 1, self.input_len + self.output_len, self.hidden_dims))

        self.pos_emb = nn.Parameter(torch.FloatTensor(1, self.input_len, self.hidden_dims))

        self.st_conv_encoder = SpatialTemporalConv(self.capacity, self.var_len, self.hidden_dims, self.input_len)
        self.st_conv_decoder = SpatialTemporalConv(self.capacity, self.var_len, self.hidden_dims, self.input_len + self.output_len)

        self.enc = Encoder(self.var_len, self.input_len, self.output_len, self.hidden_dims, self.n_heads, self.num_layers, self.dropout)
        self.dec = Decoder(self.var_len, self.input_len, self.output_len, self.hidden_dims, self.n_heads, 1, self.dropout, self.capacity)

        self.pred_nn = nn.Linear(self.hidden_dims, self.capacity)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch_x, x_cat, adj):
        bz, id_len, input_len, var_len = batch_x.shape
        # print(batch_x.shape)
        adj = torch.Tensor(adj).to(batch_x.device)
        batch_x = batch_x.permute(0, 2, 1, 3)

        batch_pred_trend = torch.mean(batch_x, 1, keepdim=True)[:, :, :, -1]
        batch_pred_trend = batch_pred_trend.repeat(1, self.output_len, 1)
        batch_pred_trend = torch.concat( [self.decomp(batch_x[:, :, :, -1])[0], batch_pred_trend], 1)

        batch_x = batch_x.reshape(bz, input_len, var_len * id_len)
        _, season_init = self.decomp(batch_x)

        batch_pred_season = torch.zeros([bz, self.output_len, var_len * id_len], device=batch_x.device)
        batch_pred_season = torch.concat([season_init, batch_pred_season], dim=1)

        batch_x = self.st_conv_encoder(batch_x, adj) + self.pos_emb

        batch_pred_season = self.st_conv_decoder( batch_pred_season, adj) + self.pos_dec_emb
        # print(batch_x.shape, batch_pred_season.shape, batch_pred_trend.shape)
        batch_x = self.enc(batch_x)

        batch_x_pred, batch_x_trends = self.dec(batch_pred_season, batch_pred_trend, batch_x)
        batch_x_pred = self.pred_nn(batch_x_pred)

        pred_y = batch_x_pred + batch_x_trends
        pred_y = pred_y.permute(0, 2, 1)[:, :, -self.output_len:]
        return pred_y


