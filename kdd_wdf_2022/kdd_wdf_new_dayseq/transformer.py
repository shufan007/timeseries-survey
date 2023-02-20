# Arxiv Link https://arxiv.org/pdf/1907.00235.pdf

import numpy as np
import torch
import torch.nn as nn
import math
import copy
from torch.nn.parameter import Parameter

activation_dict = {"ReLU": torch.nn.ReLU(), "Softplus": torch.nn.Softplus(), "Softmax": torch.nn.Softmax}


class logTransformer(nn.Module):
    def __init__(self, settings):
        """
        Args:
            n_time_series: Number of time series present in input
            n_head: Number of heads in the MultiHeadAttention mechanism
            seq_num: The number of targets to forecast
            sub_len: sub_len of the sparse attention
            num_layer: The number of transformer blocks in the model.
            n_embd: The dimention of Position embedding and time series ID embedding
            forecast_history: The number of historical steps fed into the time series model
            dropout: The dropout for the embedding of the model.
            additional_params: Additional parameters used to initalize the attention model. Can inc
        """
        super(logTransformer, self).__init__()
        self.seq_len = settings["input_len"]
        self.horizon = settings["output_len"]
        self.n_embd = settings["hidden_dims"]
        self.input_dim = settings["var_len"] - 2
        self.output_dim = settings["var_out"]
        self.n_header = settings["nhead"]
        self.dropout = settings["dropout"]
        self.num_layers = settings["num_layers"]
        self.day_seq_len = settings["day_seq_len"]
        self.scale_att = False
        self.sub_len = 24

        #transformer for hour_sequence
        self.transformer = TransformerModel(self.input_dim, self.n_header, self.sub_len, self.num_layers, self.n_embd, self.seq_len,
                                            self.dropout, self.scale_att, self.seq_len , horizon=self.horizon)
        self.sigma1 = nn.Conv1d(self.seq_len, self.horizon, kernel_size=self.input_dim+1)
        fc_dim = self.n_embd
        #transformer for day_sequence
        self.day_transformer = TransformerModel(self.input_dim, self.n_header, 2, self.num_layers, self.n_embd, self.day_seq_len,
                                            self.dropout, self.scale_att, self.day_seq_len, horizon=self.horizon)
        self.sigma2 = nn.Conv1d(self.day_seq_len, self.horizon, kernel_size=self.input_dim + 1)
        fc_dim += self.n_embd
        # current
        # self.cat_emb = 143
        # self.emb_time = nn.Embedding(144, 143)
        # fc_dim += self.cat_emb
        self.fc = nn.Sequential(
                nn.Linear(fc_dim, 128)
                , nn.Tanh()
                , nn.Linear(128, 64)
                , nn.Tanh()
                , nn.Linear(64, 64)
                , nn.Tanh()
                , nn.Dropout(p=0.3)
                , nn.Linear(64, 1)
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.bias, 0)

    def forward(self, day_input: torch.Tensore):
        """
        Args:
            x: Tensor of dimension (batch_size, seq_len, number_of_time_series)
            series_id: Optional id of the series in the dataframe. Currently  not supported
        Returns:
            Case 1: tensor of dimension (batch_size, forecast_length)
            Case 2: GLoss sigma and mu: tuple of ((batch_size, forecast_history, 1), (batch_size, forecast_history, 1))
        """
        bs, nodes = day_input.shape[0], day_input.shape[1]

        day_input = day_input.reshape(bs * nodes, self.day_seq_len, self.input_dim)
        d_output = self.day_transformer(day_input)   # [bs, t, day_variables + n_embd]
        del day_input
        # print('day',d_output.shape)
        d_output = self.sigma2(d_output)    # [bs, t_o, n_embd]
        output = torch.cat([d_output, output], dim=-1)  # [bs, t_o,  2 * n_embd]
        del d_output
        output = output.reshape(-1, 2 * self.n_embd)

        # time_id = self.emb_time(time_id[..., 1].type(torch.int32))
        # output = torch.cat([output, time_id], dim=-1)    # [bs, 3 * n_embd ]
        output = self.fc(output).reshape(bs, nodes, -1)

        return output


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)

ACT_FNS = {
    'relu': nn.ReLU(),
    'swish': swish,
    'gelu': gelu
}


class Attention(nn.Module):
    def __init__(self, n_head, n_embd, win_len, scale, q_len, sub_len, sparse=None, attn_pdrop=0.1, resid_pdrop=0.1):
        super(Attention, self).__init__()

        if(sparse):
            print('Activate log sparse!')
            mask = self.log_mask(win_len, sub_len)
        else:
            mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)

        self.register_buffer('mask_tri', mask)
        self.n_head = n_head
        self.split_size = n_embd * self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, self.q_len)
        self.value = Conv1D(n_embd * n_head, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd * self.n_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.
        2 . Our default setting here use Local attention and Restart attention.
        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros((win_len), dtype=torch.float)
        if((win_len // sub_len) * 2 * (log_l) > index):
            mask[:(index + 1)] = 1
        else:
            while(index >= 0):
                if((index - log_l + 1) < 0):
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if((index - new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query: torch.Tensor, key, value: torch.Tensor, activation="Softmax"):
        activation = activation_dict[activation](dim=-1)
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):

        value = self.value(x)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0))
        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn = self.attn(query, key, value)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)
        return attn

class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."
    def __init__(self, n_embd, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_embd))
        self.b = nn.Parameter(torch.zeros(n_embd))
        self.e = e

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma + self.e)
        return self.g * x + self.b

class MLP(nn.Module):
    def __init__(self, n_state, n_embd, acf='gelu'):
        super(MLP, self).__init__()
        n_embd = n_embd
        self.c_fc = Conv1D(n_state, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_state)
        self.act = ACT_FNS[acf]
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        hidden1 = self.act(self.c_fc(x))
        hidden2 = self.c_proj(hidden1)
        return self.dropout(hidden2)


class Block(nn.Module):
    def __init__(self, n_head, win_len, n_embd, scale, q_len, sub_len):
        super(Block, self).__init__()
        n_embd = n_embd
        self.attn = Attention(n_head, n_embd, win_len, scale, q_len, sub_len)
        self.ln_1 = LayerNorm(n_embd)
        #self.mlp = MLP(2 * n_embd, n_embd)
        #self.ln_2 = LayerNorm(n_embd)

    def forward(self, x):
        attn = self.attn(x)
        hidden = self.ln_1(x + attn)
        #mlp = self.mlp(hidden)
        #hidden = self.ln_2(hidden + mlp)
        return hidden


class TransformerModel(nn.Module):
    """ Transformer model """

    def __init__(self, input_dim, n_head, sub_len, num_layer, n_embd,
                 forecast_history: int, dropout: float, scale_att, q_len, horizon=None):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.n_head = n_head
        self.horizon = horizon
        self.n_embd = n_embd
        self.win_len = forecast_history
        # The following is the implementation of this paragraph
        """ For positional encoding in Transformer, we use learnable position embedding.
        For covariates, following [3], we use all or part of year, month, day-of-the-week,
        hour-of-the-day, minute-of-the-hour, age and time-series-ID according to the granularities of datasets.
        age is the distance to the first observation in that time series [3]. Each of them except time series
        ID has only one dimension and is normalized to have zero mean and unit variance (if applicable).
        """
        # self.po_embed = nn.Embedding(forecast_history, n_embd)
        # self.drop_em = nn.Dropout(dropout)
        block = Block(n_head, forecast_history, n_embd + self.input_dim, scale=scale_att,
                      q_len=q_len, sub_len=sub_len)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layer)])
        # nn.init.normal_(self.po_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor):
        """Runs  forward pass of the DecoderTransformer model.
        :param x: [description]
        :type x: torch.Tensor
        :return: [description]
        :rtype: [type]
        """
        batch_size = x.size(0)
        length = x.size(1)  # (Batch_size, length, input_dim)
        embedding_sum = torch.zeros(batch_size, length, self.n_embd).to(x.device)

        # position = torch.Tensor(torch.arange(length), dtype=torch.long).to(x.device)
        # po_embedding = self.po_embed(position)
        # embedding_sum[:] = po_embedding
        x = torch.cat((x, embedding_sum), dim=2)
        for block in self.blocks:
            x = block(x)
        return x


