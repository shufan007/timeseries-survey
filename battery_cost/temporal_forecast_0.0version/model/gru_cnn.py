import torch
import torch.nn.functional as F
import torch.nn as nn

from model.utils import fully_connected_layer, GLU

class GRUCell(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GRUCell, self).__init__()
        self.hidden_dim = dim_out
        self.reset_and_update_gates = nn.Linear(dim_in+self.hidden_dim, 2*dim_out)
        self.hidden_update = nn.Linear( dim_in+self.hidden_dim, dim_out)

    def forward(self, x, state):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.reset_and_update_gates(input_and_state))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)

        hidden_temp = torch.tanh(self.hidden_update( torch.cat( (x, z * state), dim=-1)))
        hidden = r * state + (1 - r) * hidden_temp
        return hidden

    def get_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)

class TRNN(nn.Module):
    def __init__(self, dim_in, dim_out, **param):
        super(TRNN, self).__init__()
        self.seq_len = int(param.get('seq_len'))
        self.input_dim = dim_in
        self.num_layers = int(param.get("num_layers"))
        self.dim_list = [self.input_dim] + [dim_out] * self.num_layers
        self.rnn_cells = nn.ModuleList( [ GRUCell(self.dim_list[tt],
                                                     self.dim_list[tt + 1]
                                                    ) for tt in range( self.num_layers ) ] )
    def forward(self, x):
        assert x.shape[-1] == self.input_dim and x.shape[1] == self.seq_len
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = self.rnn_cells[i].get_hidden_state(x.shape[0])
            inner_states = []
            #seq_to_seq
            for t in range(seq_length):
                state = self.rnn_cells[i](current_inputs[:, t, :], state)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs

class GRU(nn.Module):
    def __init__(self, **param):
        super(GRU, self).__init__()
        #self.input_dim = int(param.get('input_dim'))
        self.input_dim = 1
        self.hidden_dim = int(param.get("rnn_units"))
        self.output_dim = int(param.get("output_dim"))
        self.seq_len = int(param.get("seq_len"))
        self.horizon = int(param.get('horizon'))
        self.encoder = TRNN(self.input_dim, self.hidden_dim, **param)

        #self.gated_activation = GLU(self.hidden_dim)
        self.fc = fully_connected_layer(self.horizon * self.hidden_dim, [128, 128, 64, 64], 1,
                                        activation_type='tanh', norm_layer=True, last_dropout=0.5)

        self._reset_parameters

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, source):
        """"
        Args:
            source: Required. Tensor of dimension (batch_size, seq_len, number_of_features)
        Returns:
            output: tensor of dimension (batch_size, forecast_length)
        """

        source = source[..., -self.input_dim:]  # shape of x: (B, T, D)
        output = self.encoder(source)  # B, T, hidden

        #output = self.gated_activation(output)
        output = self.fc(output[:, -self.horizon:, :].reshape(-1, self.horizon * self.hidden_dim)) # B, T, hidden_tmp

        assert output.shape[0] == source.shape[0]
        return output