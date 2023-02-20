import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import fully_connected_layer, GLU
from model.embedding import _conv_embedding, DataEmbedding, PositionalEmbedding


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
    def __init__(self, ** param):
        super(TransformerTimeSeries, self).__init__()
        self.seq_len = int(param.get('seq_len'))
        self.horizon = int(param.get('horizon'))
        self.categoical_variables = int(param.get('categoical_variables'))
        self.input_dim = int(param.get('input_dim'))
        self.encode_dim = int(param.get('encode_dim'))
        self.output_dim = int(param.get('output_dim'))
        self.emb_dim = int(param.get('emb_dim'))
        self.n_header = int(param.get('n_header'))
        self.d_model = int(param.get('d_model'))
        self.num_layers = int(param.get('num_layers'))
        self.dim_feedforward = int(param.get('dim_feedforward'))
        # categoical_variables deals
        # self.categoical_embedding = nn.ModuleList([nn.Embedding(64, self.emb_dim) for _ in range(self.categoical_variables)] )
        # embedding
        self.enc_embedding = DataEmbedding(self.encode_dim, self.d_model, dropout=0.1)
        self.dec_embedding = DataEmbedding(self.input_dim-self.encode_dim, self.d_model, dropout=0.1)
        # transformer layers
        self.transformer = nn.Transformer(self.d_model, self.n_header, num_decoder_layers=self.num_layers, num_encoder_layers=self.num_layers,
                                          dim_feedforward=self.dim_feedforward, dropout=0.1, batch_first=True)

        # output layer
        #self.gated_activation = GLU(self.d_model)
        self.fc = fully_connected_layer(self.d_model, [self.d_model, 128, 128, 64, 64], 1, activation_type='tanh', norm_layer=True, last_dropout=0.5)
        #self.end_predictor = nn.Conv1d(self.horizon, self.horizon, kernel_size=self.d_model)
        self._reset_parameters
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    def categoical_emb(self, input):
        time_varying_categoical_vectors = []
        for i in range(self.categoical_variables):
            emb = self.categoical_embedding[i](input[:, :, i].view(input.size(0), -1, 1).long())
            time_varying_categoical_vectors.append(emb)
        categoical_embedding = torch.cat(time_varying_categoical_vectors,
                                                      dim=2)  # [bs, seq_len, categoical_variables*embedding_dim]
        return categoical_embedding.view(input.size(0), -1, self.categoical_variables * self.emb_dim)
    def forward(self, input):
        """"
        Args:
            input: Required. Tensor of dimension (batch_size, seq_len, number_of_features)
        Returns:
            output: tensor of dimension (batch_size, forecast_length)
        """
        input = input[:, -self.seq_len:, -self.input_dim:]
        en_input = input[:, :self.seq_len - self.horizon, :self.encode_dim]
        de_input = input[:, -self.horizon:, self.encode_dim:]
        input_embedding = self.enc_embedding(en_input)
        tgt = self.dec_embedding(de_input)

        output = self.transformer(input_embedding, tgt)

        # fully connected
        output = self.fc(output)
        output = output[:, -1, :]
        assert output.shape[0] == input.shape[0]

        return output


