import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import fully_connected_layer, GLU
from model.embedding import _conv_embedding


class TransformerTimeSeries(nn.Module):
    """
    Time Series application of transformers

    _conv_embedding parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel

    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)

    fc parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector

    GLU parameters:
        channels: the output size of fc-hidden layers

    """

    def __init__(self, **param):
        super(TransformerTimeSeries, self).__init__()
        self.seq_len = int(param.get('seq_len'))
        self.horizon = int(param.get('horizon'))
        self.input_dim = int(param.get('input_dim'))
        self.output_dim = int(param.get('output_dim'))
        self.n_header = int(param.get('n_header'))
        self.d_model = int(param.get('d_model'))
        self.num_layers = int(param.get('num_layers'))
        self.kernel_size = int(param.get('kernel_size'))
        self.conv_emb = True
        if self.conv_emb:
            self.en_embedding = _conv_embedding(self.input_dim, self.d_model, 3)
        else:
            self.en_embedding = nn.Linear(self.input_dim, self.d_model)
        # transformer layers
        self.encode_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_header, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encode_layer, num_layers=self.num_layers)
        # output layer
        # self.gated_activation = GLU(self.d_model)
        self.fc = fully_connected_layer(self.d_model, [self.d_model, 128, 128, 64, 64], 1, activation_type='tanh', norm_layer = True, last_dropout=0.5)
        self.end_predictor = nn.Conv1d(self.horizon, self.output_dim, kernel_size = self.d_model)

        self._reset_parameters

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, input):
        """"
        Args:
            input: Required. Tensor of dimension (batch_size, seq_len, number_of_features)
        Returns:
            output: tensor of dimension (batch_size, forecast_length)
        """
        input = input[..., -self.input_dim:]
        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        if self.conv_emb:
            en_embedding = self.en_embedding(input.permute(0, 2, 1)).permute(0, 2, 1) # [B, T_in, F]->[B, T_in, h]
        else:
            en_embedding = self.en_embedding(input)

        output = self.transformer_encoder(en_embedding) #[B, T_in, h]

        # output = self.gated_activation(output)
        # fully connected
        # output = self.fc(output[:, -1, :])
        output = self.end_predictor(output[:, -self.horizon:, :]).reshape(-1, 1)

        assert output.shape[0] == input.shape[0]

        return output