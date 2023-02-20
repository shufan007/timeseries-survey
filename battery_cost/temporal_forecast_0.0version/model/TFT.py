from torch import nn
import math
import torch
from model.utils import GLU
from model.embedding import PositionalEmbedding

class QuantileLoss(nn.Module):
    ## From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

    def __init__(self, quantiles):
        ##takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class TimeDistributed(nn.Module):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_state_size, output_size, dropout, hidden_context_size=None,
                 batch_first=False):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size = hidden_state_size
        self.dropout = dropout

        if self.input_size != self.output_size:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_size, self.output_size))

        self.fc1 = TimeDistributed(nn.Linear(self.input_size, self.hidden_state_size), batch_first=batch_first)
        self.elu1 = nn.ELU()

        if self.hidden_context_size is not None:
            self.context = TimeDistributed(nn.Linear(self.hidden_context_size, self.hidden_state_size),
                                           batch_first=batch_first)

        self.fc2 = TimeDistributed(nn.Linear(self.hidden_state_size, self.output_size), batch_first=batch_first)
        self.elu2 = nn.ELU()

        self.dropout = nn.Dropout(self.dropout)
        self.bn = TimeDistributed(nn.BatchNorm1d(self.output_size), batch_first=batch_first)
        self.gate = TimeDistributed(GLU(self.output_size), batch_first=batch_first)

    def forward(self, x, context=None):

        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu1(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x + residual
        x = self.bn(x)

        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context=None):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context = context

        if self.context is not None:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs * self.input_size, self.hidden_size,
                                                      self.num_inputs, self.dropout, self.context * self.input_size)
        else:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs * self.input_size, self.hidden_size,
                                                      self.num_inputs, self.dropout)

        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size, self.dropout))

        self.softmax = nn.Softmax()

    def forward(self, embedding, context=None):
        if context is not None:
            sparse_weights = self.flattened_grn(embedding, context)
        else:
            sparse_weights = self.flattened_grn(embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        var_outputs = []
        for i in range(self.num_inputs):
            ##select slice of embedding belonging to a single input
            var_outputs.append(
                self.single_variable_grns[i](embedding[:, :, (i * self.input_size): (i + 1) * self.input_size]))

        var_outputs = torch.stack(var_outputs, axis=-1)

        outputs = var_outputs * sparse_weights

        outputs = outputs.sum(axis=-1)

        return outputs


class TFT(nn.Module):
    def __init__(self, **param):
        super(TFT, self).__init__()
        self.current = bool(param.get('current'))
        self.encode_length = int(param.get('encode_length'))
        self.categoical_variables = int(param.get('categoical_variables'))
        self.variables_encoder = int(param.get('variables_encoder'))
        self.variables_decoder = int(param.get('variables_decoder'))
        self.num_to_mask = int(param.get('num_masked_series'))
        self.hidden_size = int(param.get('lstm_hidden_dimension'))
        self.lstm_layers = int(param.get('lstm_layers'))
        self.dropout = float(param.get('dropout'))
        self.static_embedding_vocab_sizes = int(param.get('static_embedding_vocab_sizes'))
        self.embedding_vocab_sizes = int(param.get('embedding_vocab_sizes'))
        self.embedding_dim = int(param.get('embedding_dim'))
        self.attn_heads = int(param.get('attn_heads'))
        self.output_dim = int(param.get('output_dim'))
        self.seq_length = int(param.get('seq_len'))
        if self.current:
            self.static_variables = int(param.get('static_variables'))
            self.current_variables = int(param.get('current_variables'))
            self.static_embedding = TimeDistributed(nn.Embedding(self.static_embedding_vocab_sizes, self.embedding_dim),
                                                    batch_first=True)
            self.current_linear = TimeDistributed(nn.Linear(self.current_variables, self.current_variables * self.embedding_dim),
                                                  batch_first=True)
            self.current_num_emb = self.current_variables - self.static_variables
        else:
            self.current_num_emb = 0
            self.static_variables = 0
        if self.static_variables == 0 :  static_variables = None
        if self.categoical_variables > 0:
            self.time_varying_embedding =TimeDistributed(nn.Embedding(self.embedding_vocab_sizes, self.embedding_dim), batch_first=True)

        self.time_varying_linear_layers = nn.ModuleList()
        for i in range(self.variables_encoder):
            emb = TimeDistributed(nn.Linear(1, self.embedding_dim), batch_first=True)
            self.time_varying_linear_layers.append(emb)
        self.num_encode_inputs = self.variables_encoder - self.variables_decoder + self.categoical_variables + self.current_num_emb
        self.num_decode_inputs = self.variables_decoder + self.categoical_variables + self.current_num_emb
        self.encoder_variable_selection = VariableSelectionNetwork(self.embedding_dim,
                                                                   self.num_encode_inputs,
                                                                   self.hidden_size,
                                                                   self.dropout,
                                                                   static_variables)

        self.decoder_variable_selection = VariableSelectionNetwork(self.embedding_dim,
                                                                   self.num_decode_inputs,
                                                                   self.hidden_size,
                                                                   self.dropout,
                                                                   static_variables)

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.lstm_layers,
                                    dropout=self.dropout)

        self.lstm_decoder = nn.LSTM(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.lstm_layers,
                                    dropout=self.dropout)

        self.post_lstm_gate = TimeDistributed(GLU(self.hidden_size))
        self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size))
        self.position_embedding = PositionalEmbedding(self.hidden_size, self.encode_length)
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.attn_heads)
        self.post_attn_gate = TimeDistributed(GLU(self.hidden_size))

        self.post_attn_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size))
        self.pos_wise_ff = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout)

        self.pre_output_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size))
        self.pre_output_gate = TimeDistributed(GLU(self.hidden_size))
        self.output_layer = TimeDistributed(nn.Linear(self.hidden_size, self.output_dim), batch_first=True)

        self._reset_parameters

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    def init_hidden(self, batch_size):
        return torch.zeros(self.lstm_layers, batch_size, self.hidden_size)

    def apply_embedding(self, x, current_embedding, apply_masking):
        ###x should have dimensions (batch_size, timesteps, input_size)
        ## Apply masking is used to mask variables that should not be accessed after the encoding steps
        ## return: embeddings.size(2) = time_varying_categoical_variables + static_variables * embedding_dim + time_varying_real_variables_encoder
        #                               = 5 + 5*8 + 48 + 4
        ##Time-varying categorical embeddings (ie hour)
        if self.categoical_variables > 0:
            time_varying_categoical_vectors = []
            for i in range(self.categoical_variables):
                emb = self.time_varying_embedding_layers[i](
                    x[:, :, i].view(x.size(0), -1, 1).long())
                time_varying_categoical_vectors.append(emb)
            time_varying_categoical_embedding = torch.cat(time_varying_categoical_vectors, dim=2) # [bs, seq_len, categoical_variables*embedding_dim]

        # Time-varying real embeddings
        if apply_masking: # decoder
            time_varying_real_vectors = []
            for i in range(self.variables_decoder):
                emb = self.time_varying_linear_layers[i + self.num_to_mask ](
                    x[:, :, i + self.num_to_mask ].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2) # [bs, seq_len, categoical_decoder*embedding_dim]

        else: # encoder
            time_varying_real_vectors = []
            for i in range(self.variables_encoder - self.variables_decoder):
                emb = self.time_varying_linear_layers[i](x[:, :, i + self.categoical_variables].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2) # [bs, seq_len, categoical_encoder*embedding_dim]

        ##repeat static_embedding for all timesteps
        if self.current:
            static_embedding = torch.cat(time_varying_real_embedding.size(1) * [current_embedding])
            static_embedding = static_embedding.view(time_varying_real_embedding.size(0),
                                                 time_varying_real_embedding.size(1), -1)
            if self.categoical_variables > 0:
                ##concatenate all embeddings
                embeddings = torch.cat([static_embedding, time_varying_categoical_embedding, time_varying_real_embedding], dim=2)
                return embeddings.permute(1, 0, 2)
            else:
                embeddings = torch.cat([static_embedding, time_varying_real_embedding], dim=2)
                return embeddings.permute(1, 0, 2)
        if self.categoical_variables > 0:
            embeddings = torch.cat([ time_varying_categoical_embedding, time_varying_real_embedding],
                                   dim=2)
            return embeddings.permute(1, 0, 2)
        else:
            return time_varying_real_embedding.permute(1, 0, 2)

    def encode(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(1)).to(x.device)
        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))

        return output, hidden

    def decode(self, x, hidden=None):

        if hidden is None:
            hidden = self.init_hidden(x.size(1)).to(x.device)

        output, (hidden, cell) = self.lstm_decoder(x, (hidden, hidden))

        return output, hidden

    def forward(self, seq_feature, current_feature=None, day_feature=None):
        """"
        Args:
            seq_feature: Required. Tensor of dimension (batch_size, seq_len, number_of_features)
            current_feature: Optional.Tensor of dimension (batch_size, number_of_features)
            day_feature:Optional.Tensor of dimension (batch_size, day_num, seq_len_per_day, number_of_features)
        Returns:
            output: tensor of dimension (batch_size, forecast_length)
        """
        if self.current:

            if not self.static_variables:
                static_variables = 0
            current_embedding = self.current_linear(current_feature[:, static_variables - self.current_variables:])
            if self.static_variables:
                static_embedding = self.static_embedding(current_feature[:, :self.static_variables].long()) # [bs, embedding_dim * static_variables]
                current_embedding = torch.cat([current_embedding, static_embedding.view(current_embedding.size(0), -1)], dim=1)
        seq_feature = seq_feature[:, -self.seq_length:, -self.variables_encoder:]
        ##Embedding and variable selection
        embeddings_encoder = self.apply_embedding(seq_feature[:, :self.encode_length, :], current_embedding, apply_masking=False)  # [t, b, f]
        embeddings_decoder = self.apply_embedding(seq_feature[:, self.encode_length:, :], current_embedding, apply_masking=True)
        #print(current_embedding.shape, embeddings_encoder.shape, embeddings_decoder.shape)

        if self.current and self.static_variables:
            embeddings_encoder = self.encoder_variable_selection(
                embeddings_encoder[:, :, (self.embedding_dim * self.static_variables):],
                embeddings_encoder[:, :, :(self.embedding_dim * self.static_variables)])
            embeddings_decoder = self.decoder_variable_selection(
                embeddings_decoder[:, :, (self.embedding_dim * self.static_variables):],
                embeddings_decoder[:, :, :(self.embedding_dim * self.static_variables)])
        else:
            embeddings_encoder = self.encoder_variable_selection(embeddings_encoder)
            embeddings_decoder = self.decoder_variable_selection(embeddings_decoder)

        #print(embeddings_encoder.shape, embeddings_decoder.shape)
        pe = torch.zeros(self.seq_length, 1, embeddings_encoder.shape[2], device=embeddings_encoder.device)
        pe = (self.position_embedding(pe) + pe)

        embeddings_encoder = embeddings_encoder + pe[:self.encode_length, :, :]
        embeddings_decoder = embeddings_decoder + pe[self.encode_length:, :, -embeddings_decoder.shape[2]:]

        ##LSTM
        lstm_input = torch.cat([embeddings_encoder, embeddings_decoder], dim=0)
        encoder_output, hidden = self.encode(embeddings_encoder)
        decoder_output, _ = self.decode(embeddings_decoder, hidden)
        lstm_output = torch.cat([encoder_output, decoder_output], dim=0)

        ##skip connection over lstm
        lstm_output = self.post_lstm_gate(lstm_output + lstm_input) #glu

        ##skip connection over lstm
        attn_input = self.post_lstm_norm(lstm_output)

        ##Attention
        attn_output, _ = self.multihead_attn(attn_input[self.encode_length:, :, :],
                                                               attn_input[:self.encode_length, :, :],
                                                               attn_input[:self.encode_length, :, :])

        ##skip connection over attention
        attn_output = self.post_attn_gate(attn_output) + attn_input[self.encode_length:, :, :]
        attn_output = self.post_attn_norm(attn_output)

        output = self.pos_wise_ff(attn_output)  # [self.encode_length:,:,:])

        ##skip connection over Decoder
        output = self.pre_output_gate(output) + lstm_output[self.encode_length:, :, :]

        # Final output layers
        #output = self.pre_output_norm(output)
        output = self.output_layer(output.view(seq_feature.size(0), -1, self.hidden_size)).view(-1, self.output_dim)
        assert output.shape[0] == seq_feature.shape[0]

        return output