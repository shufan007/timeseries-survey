import torch
import torch.nn as nn
import torch.nn.functional as F

class ChebyNetcell(nn.Module):
    """ Chebyshev Network, see reference below for more information
        Defferrard, M., Bresson, X. and Vandergheynst, P., 2016.
        Convolutional neural networks on graphs with fast localized spectral filtering. In NIPS.
    """
    def __init__(self, dim_in, dim_out):
        super(ChebyNetcell, self).__init__()
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.num_edgetype = 1
        #self.polynomial_order = max(self.long_diffusion_dist)
        self.polynomial_order = 3
        self.dropout = 0.1
        self.filter = nn.Linear(dim_in * (self.polynomial_order + self.num_edgetype), dim_out)

    def forward(self, node_feat, L):
        """
          shape parameters:
            batch size = B
            input dim = D_in
            output dim = D_out
            the number of nodes  = N
            number of edge types = E

          Args:
            node_feat: long tensor, shape B  X N X D_in
            L: float tensor, shape N X N X (E + 1)
            state: long tensor, shape B  X N X D_out
        """
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = node_feat
        # propagation
        # Note: we assume adj = 2 * L / lambda_max - I
        state_scale = [None] * (self.polynomial_order + 1)
        state_scale[-1] = state
        state_scale[0] = torch.einsum('nn,bnd->bnd', L, state)
        for kk in range(1, self.polynomial_order):
            state_scale[kk] = 2.0 * torch.einsum('nn, bnd->bnd',
                L, state_scale[kk - 1]) - state_scale[kk - 2]

        msg = []
        for ii in range(0, self.num_edgetype):
            msg += [torch.einsum('nn,bnd->bnd', L, state)]  # shape: B X N X D

        msg = torch.cat(msg + state_scale, dim=2).view(num_node * batch_size, -1)
        state = F.relu(self.filter(msg)).view(batch_size, num_node, -1)
        state = F.dropout(state, self.dropout, training=self.training)
        return state

class RNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, num_layers):
        super(RNN, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        emb_dim = [dim_out] * num_layers
        self.dim_list = [dim_in] + emb_dim
        self.rnn_cells = nn.ModuleList([ChebyNetcell(self.dim_list[tt],
                                                     self.dim_list[tt + 1])
                                        for tt in range(self.num_layers)])

    def forward(self, x, L):
        """
          shape parameters:
            input dim = D_in
            output dim = D_out
            the number of nodes  = N
            number of edge types = E

          Args:
            node_feat: long tensor, shape B  X N X D_in
            L: float tensor, shape N X N X (E + 1)
            state: long tensor, shape B  X N X D_out
        """
        seq_length = x.shape[1]
        current_inputs = x
        L = torch.Tensor(L).to(x.device)
        for i in range(self.num_layers):
            inner_states = []
            for t in range(seq_length):
                state = self.rnn_cells[i](current_inputs[:, t, :, :], L)
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs


class ChebyNet(nn.Module):
    def __init__(self, settings):
        super(ChebyNet, self).__init__()
        self.num_node = settings["capacity"]
        self.input_dim = settings["var_len"] - 2
        self.hidden_dim = settings["hidden_dims"]
        self.output_dim = settings["var_out"]
        self.horizon = settings["output_len"]
        self.seq_len = settings["input_len"]
        self.day_seq_len = settings["day_seq_len"]
        self.num_layers = settings["num_layers"]
        self.gru_encode = nn.GRU(input_size=self.input_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=self.num_layers,
                          batch_first=True)
        self.day_encoder = RNN(self.num_node, self.input_dim, self.hidden_dim, self.num_layers )
        # predictor
        self.end_conv = nn.Conv2d(self.day_seq_len + self.seq_len, self.horizon, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, x_input, time_id, day_seq, adj):
        """
          shape parameters:
            batch size = B
            length of history data: T_in
            length of history data: T_in
            input dim = D_in
            output dim = D_out = 1
            the number of nodes  = N

          Args:
            node_feat: long tensor, shape B, T_in, N, D_in
            output: long tensor, shape B, T_out, N, D_out
        """
        bs = x_input.shape[0]
        output = self.day_encoder(day_seq.permute(0, 2, 1, 3), adj)  # B, T, N, hidden
        del day_seq, adj

        x_input = x_input[..., :self.input_dim]  # shape of x: (B, T_in, N, D_in)
        x_output, _ = self.gru_encode(x_input.reshape(bs * self.num_node, self.seq_len, self.input_dim))
        x_output = x_output.reshape(bs, self.num_node, self.seq_len, self.hidden_dim)
        del x_input, time_id

        # CNN based predictor
        output = torch.cat((x_output.permute(0, 2, 1, 3), output), dim=1)
        output = self.end_conv(output)  # B, T_out * D_out, N, 1
        output = output.reshape(bs, self.horizon, self.num_node)
        output = F.relu(output.permute(0, 2, 1))  # B, N, T_out
        return output