import math
import torch
import torch.nn as nn

class MultiHeadSelectAttention(nn.Module):
    def __init__(self, input_size, n_head, project_size, value_input, value_size):
        super(MultiHeadSelectAttention, self).__init__()
        self.n_head = n_head
        self.key_size = self.query_size = project_size // n_head
        self.trans_q = nn.Linear(input_size, project_size, bias=False)
        self.trans_k = nn.Linear(input_size, project_size, bias=False)
        self.trans_v = nn.Linear(value_input, value_size * n_head, bias=False)
        self.value_size = value_size
        self.sqrt_k = math.sqrt(self.key_size)
        self.layer_norm = nn.LayerNorm(value_size * n_head)

    def forward(self, query_input, key_input, value_input):
        batch_size, seq_len, _ = query_input.shape
        query = self.trans_q(query_input).transpose(2, 1).reshape(batch_size * self.n_head, seq_len, -1)
        key = self.trans_k(key_input).reshape(batch_size * self.n_head, -1, 1)
        tmp_v = self.trans_v(value_input).transpose(2, 1).reshape(batch_size * self.n_head, seq_len, -1)
        # b * n * seq_len
        dot = torch.softmax(torch.bmm(query, key).squeeze(1) / self.sqrt_k, 1)
        value = dot.expand(batch_size * self.n_head, seq_len, self.value_size) * tmp_v
        skip_connect = value.view(batch_size, self.n_head, seq_len, -1).transpose(2, 1).reshape(batch_size, seq_len, -1)
        res = self.layer_norm(skip_connect)
        return res


class GruModel(nn.Module):
    def __init__(self):
        super(GruModel, self).__init__()
        cate_shape = 78
        cate_size = 5
        seq_nume_size = 48  # seq_nume_size
        flat_nume_size = 44  # flat_nume_size
        cat_hidden = 5
        seq_hidden = 64  # seq_hidden
        extra_flat_dim = 274
        static_size = 24
        mha_attn_hidden = 32
        n_head = 4
        self.cat_emb = nn.Embedding(cate_shape, cat_hidden, padding_idx=cate_shape - 1)
        self.seq_hidden = seq_hidden

        seq_in = cate_size * cat_hidden + seq_nume_size
        static_input_size = static_size + cate_size * cat_hidden
        gru_in = n_head * mha_attn_hidden + seq_in

        self.h_gru = nn.GRU(gru_in, seq_hidden, 1, batch_first=True)
        self.h_attn = MultiHeadSelectAttention(static_input_size, n_head, n_head * mha_attn_hidden, seq_in,
                                               mha_attn_hidden)
        self.d_gru = nn.GRU(gru_in, seq_hidden, 1, batch_first=True)
        self.d_attn = MultiHeadSelectAttention(static_input_size, n_head, n_head * mha_attn_hidden, seq_in,
                                               mha_attn_hidden)

        flat_in = cate_size * cat_hidden + flat_nume_size + extra_flat_dim + seq_hidden * 2
        self.PRETRAIN_MODEL_FILE = '/nfs/volume-807-1/houmin/gru/15_train'
        self.fc = nn.Sequential(
            nn.Linear(flat_in, 256)
            , nn.BatchNorm1d(256)
            , nn.Tanh()
            , nn.Linear(256, 256)
            , nn.BatchNorm1d(256)
            , nn.Tanh()
            , nn.Linear(256, 128)
            , nn.BatchNorm1d(128)
            , nn.Tanh()
            , nn.Linear(128, 128)
            , nn.BatchNorm1d(128)
            , nn.Tanh()
            , nn.Linear(128, 64)
            , nn.BatchNorm1d(64)
            , nn.Tanh()
            , nn.Linear(64, 64)
            , nn.BatchNorm1d(64)
            , nn.Tanh()
            , nn.Dropout(p=0.5)
            , nn.Linear(64, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.seq_hidden)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, batch_flat_int, batch_flat_float, batch_flat_dense,
                batch_h_int, batch_h_float, batch_h_label,
                batch_seq_int, batch_seq_float, batch_seq_label):

        batch_size = batch_flat_int.shape[0]
        h_seq_len = batch_h_int.shape[1]
        seq_len = batch_seq_int.shape[1]

        flat_emb = self.cat_emb(batch_flat_int).view(batch_size, -1)
        h_seq_emb = self.cat_emb(batch_h_int).view(batch_size, h_seq_len, -1)
        seq_emb = self.cat_emb(batch_seq_int).view(batch_size, seq_len, -1)

        flat_static = torch.cat([flat_emb, batch_flat_float[:, :24]], axis=-1)
        h_seq_static = torch.cat([h_seq_emb, batch_h_float[:, :, :24]], axis=-1)
        seq_static = torch.cat([seq_emb, batch_seq_float[:, :, :24]], axis=-1)

        h_seq_value = torch.cat([h_seq_emb, batch_h_float, batch_h_label], axis=-1)
        seq_value = torch.cat([seq_emb, batch_seq_float, batch_seq_label], axis=-1)

        h_seq_attn_out = self.h_attn(h_seq_static, flat_static, h_seq_value)
        seq_attn_out = self.d_attn(seq_static, flat_static, seq_value)
        h_gru_in = torch.cat([h_seq_value, h_seq_attn_out], axis=-1)
        d_gru_in = torch.cat([seq_value, seq_attn_out], axis=-1)
        h_out, h_hn = self.h_gru(h_gru_in)
        d_out, d_hn = self.d_gru(d_gru_in)

        flat_in = torch.cat([flat_emb, batch_flat_float, batch_flat_dense, h_hn.squeeze(0), d_hn.squeeze(0)], axis=1)
        res = self.fc(flat_in)
        return res

