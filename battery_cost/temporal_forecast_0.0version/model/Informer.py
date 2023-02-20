"""
copyright:https://app.codecov.io/gh/AIStream-Peelout/flow-forecast/blob/master/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.AttentionLayer import FullAttention, ProbAttention, AttentionLayer
from model.embedding import DataEmbedding
from model._variables_block import Current_variables_block, MultiProSelectVariables

class Informer(nn.Module):
    def __init__(self, **param):
        """ This is based on the implementation of the Informer available from the original authors
            https://github.com/zhouhaoyi/Informer2020. We have done some minimal refactoring, but
            the core code remains the same. Additionally, we have added a few more options to the code

        :param n_time_series: The number of time series present in the multivariate forecasting problem.
        :type n_time_series: int
        :param dec_in: The input size to the decoder (e.g. the number of time series passed to the decoder)
        :type dec_in: int
        :param c_out: The output dimension of the model (usually will be the number of variables you are forecasting).
        :type c_out:  int
        :param seq_len: The number of historical time steps to pass into the model.
        :type seq_len: int
        :param label_len: The length of the label sequence passed into the decoder (n_time_steps not used forecasted)
        :type label_len: int
        :param out_len: The predicted number of time steps.
        :type out_len: int
        :param factor: The multiplicative factor in the probablistic attention mechanism, defaults to 5
        :type factor: int, optional
        :param d_model: The embedding dimension of the model, defaults to 512
        :type d_model: int, optional
        :param n_heads: The number of heads in the multi-head attention mechanism , defaults to 8
        :type n_heads: int, optional
        :param e_layers: The number of layers in the encoder, defaults to 3
        :type e_layers: int, optional
        :param d_layers: The number of layers in the decoder, defaults to 2
        :type d_layers: int, optional
        :param d_ff: The dimension of the forward pass, defaults to 512
        :type d_ff: int, optional
        :param dropout: Whether to use dropout, defaults to 0.0
        :type dropout: float, optional
        :param attn: The type of the attention mechanism either 'prob' or 'full', defaults to 'prob'
        :type attn: str, optional
        :param embed: Whether to use class: `FixedEmbedding` or `torch.nn.Embbeding` , defaults to 'fixed'
        :type embed: str, optional
        :param temp_depth: The temporal depth (e.g year, month, day, weekday, etc), defaults to 4
        :type data: int, optional
        :param activation: The activation function, defaults to 'gelu'
        :type activation: str, optional
        :param device: The device the model uses, defaults to torch.device('cuda:0')
        :type device: str, optional
        """
        super(Informer, self).__init__()
        self.current = bool(param.get('current'))
        self.day = bool(param.get('day'))
        self.seq_len = int(param.get('seq_len'))
        self.horizon = int(param.get('horizon'))
        self.input_dim = int(param.get('input_dim'))
        self.encode_dim = int(param.get('encode_dim'))
        self.output_dim = int(param.get('output_dim'))
        self.n_header = int(param.get('n_header'))
        self.d_model = int(param.get('d_model'))
        self.num_layers = int(param.get('num_layers'))
        self.dim_feedforward = int(param.get('dim_feedforward'))
        self.activation = 'gelu'
        self.attn = 'prob'
        factor = 3
        encode_dim = self.encode_dim
        decode_dim = self.input_dim - self.encode_dim

        # Encoding
        self.enc_embedding = DataEmbedding(encode_dim, self.d_model,  dropout=0.2)
        self.dec_embedding = DataEmbedding(decode_dim, self.d_model,  dropout=0.2)
        # Attention
        Attn = ProbAttention if self.attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=0.5),
                                   self.d_model, self.n_header),
                    self.d_model,
                    self.dim_feedforward,
                    dropout=0.5,
                    activation=self.activation
                ) for b in range(self.num_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for b in range(self.num_layers - 1)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=0.5),
                                   self.d_model, self.n_header),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=0.5),
                                   self.d_model, self.n_header),
                    self.d_model,
                    self.dim_feedforward,
                    dropout=0.5,
                    activation=self.activation,
                )
                for c in range(self.num_layers-1)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.predictor = nn.Linear(self.d_model, self.output_dim)

    def forward(self, input: torch.Tensor, current=None, seq2_input=None):
        """"
        Args:
            input: Required. Tensor of dimension (batch_size, seq_len, number_of_features)
            current: Optional.Tensor of dimension (batch_size, number_of_features)
            seq2_input:Optional.Tensor of dimension (batch_size, day_num, seq_len_per_day, number_of_features)
        Returns:
            output: tensor of dimension (batch_size, forecast_length)
        """
        # informer for hour_sequence
        input = input[:, -self.seq_len:, -self.input_dim:]
        enc_input = input[..., :self.encode_dim]
        dec_input = input[:, -self.horizon:, self.encode_dim:]
        enc_out = self.enc_embedding(enc_input)
        enc_out = self.encoder(enc_out)

        dec_out = self.dec_embedding(dec_input)
        dec_out = self.decoder(dec_out, enc_out)

        output = self.predictor(dec_out)
        output = output[:, -1, -self.output_dim:]

        return output


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """[summary]

        :param attention: [description]
        :type attention: [type]
        :param d_model: [description]
        :type d_model: [type]
        :param d_ff: [description], defaults to None
        :type d_ff: [type], optional
        :param dropout: [description], defaults to 0.1
        :type dropout: float, optional
        :param activation: [description], defaults to "relu"
        :type activation: str, optional
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask=attn_mask
        ))

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None) -> torch.Tensor:
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        ))
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        ))

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x