

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import TokenEmbedding, DataEmbedding_wo_pos, AutoCorrelation, AutoCorrelationLayer, my_Layernorm


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, settings):
        super(Autoformer, self).__init__()
        self.seq_len = settings["input_len"]
        self.pred_len = settings["output_len"]
        self.input_dim = settings["var_len"]
        self.out_dim = settings["var_out"]
        self.num_layers = settings["num_layers"]
        self.dropout = settings["dropout"]
        self.n_heads = settings["nhead"]
        self.day_seq_len = settings["day_seq_len"]

        # static param
        self.d_model = 64
        self.d_ff = self.d_model * 2
        self.moving_avg = 25
        self.activation = 'gelu'
        kernel_size = 3
        factor = 0.8
        self.decomp = series_decomp(kernel_size)
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = TokenEmbedding(self.input_dim - 2, self.d_model)
        self.dec_embedding = TokenEmbedding(self.out_dim , self.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=self.dropout, output_attention=True),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.num_layers)
            ], norm_layer=my_Layernorm(self.d_model) )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.out_dim,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(1)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.out_dim, bias=True)
        )

        self.dec_conv = nn.Linear(2 * self.pred_len, self.pred_len)

    def forward(self, x_enc):
        batch_size, nodes, d, t, f = x_enc.shape
        # decomp init
        x_enc = x_enc.reshape(batch_size * nodes, -1, f)  # [B, T_in, D]
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)  # [B, T_out, D]
        zeros = torch.zeros([batch_size * nodes, self.pred_len, self.out_dim], device=x_enc.device)  # [B, T_out, 1]
        seasonal_init, trend_init = self.decomp(x_enc)  # [B, T_in, D]   ,  [B, T_in, D]
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.pred_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.pred_len:, -self.out_dim:], zeros], dim=1)
        # trend_init and seasonal_init: [B, 2 * T_out , 1] if T_out < T_in else: [B, T_out + T_in, 1]
        dec_out = self.dec_embedding(seasonal_init)
        del mean, zeros, seasonal_init
        # enc
        enc_out = self.enc_embedding(x_enc)  # [B, T_in, d_model]
        del x_enc
        enc_out = self.encoder(enc_out)  # [B, T_in, d_model]
        # dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)  # [B, T_out + T_in , 1], [B, T_out + T_in , D]
        del enc_out
        # final
        dec_out = self.dec_conv(trend_part[..., -self.out_dim] + seasonal_part[...,-1])       #[Bn, T_out + T_in , 1] -> [Bn, T_out, 1]
        dec_out = dec_out.reshape(batch_size, nodes, self.pred_len)  # [B, N, L]
        return dec_out


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                # attns.append(attn)
            x = self.attn_layers[-1](x)
            # attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)
                # attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend

