import torch, math
import torch.nn as nn
class SelectVariables(nn.Module):
    def __init__(self, sequence_variables, current_variables, project_size, value_size):
        super(MultiProSelectVariables, self).__init__()
        self.key_size = project_size
        self.trans_seq = nn.Conv1d(sequence_variables, project_size, bias=False)
        self.trans_cur = nn.Linear(current_variables, project_size, bias=False)

        self.value_size = value_size
        self.sqrt_k = math.sqrt(self.key_size)
        self.layer_norm = nn.LayerNorm(value_size )

    def forward(self, sequence_input, current_input):
        """
        :param sequence_input: [bs, input_len, seq_variables]
        :param current_input: [bs, cur_variables]
        :return: [bs, input_len, n_pro * value_size]
        """
        batch_size, seq_len, _ = sequence_input.shape
        query = self.trans_seq(sequence_input).transpose(2, 1).reshape(batch_size, seq_len, self.key_size)
        curkey = self.trans_cur(current_input).reshape(batch_size * self.n_pro, self.key_size, 1)
        # b * n * seq_len
        dot = torch.softmax(torch.bmm(query, curkey).squeeze(1) / self.sqrt_k, 1)
        if len(dot.shape) == 2:
            dot = dot.unsqueeze(-1)
        value = dot.expand(batch_size * self.n_pro, seq_len, self.value_size)
        skip_connect = value.view(batch_size, self.n_pro, seq_len, -1).transpose(2, 1).reshape(batch_size, seq_len, -1)
        res = self.layer_norm(skip_connect)
        return res


class MultiProSelectVariables(nn.Module):
    def __init__(self, sequence_variables, current_variables, n_pro, project_size, value_size):
        super(MultiProSelectVariables, self).__init__()
        self.n_pro = n_pro
        self.key_size = project_size // n_pro
        self.trans_seq = nn.Linear(sequence_variables, project_size, bias=False)
        self.trans_cur = nn.Linear(current_variables, project_size, bias=False)

        self.value_size = value_size
        self.sqrt_k = math.sqrt(self.key_size)
        self.layer_norm = nn.LayerNorm(value_size * n_pro)

    def forward(self, sequence_input, current_input):
        """

        :param sequence_input: [bs, input_len, seq_variables]
        :param current_input: [bs, cur_variables]
        :return: [bs, input_len, n_pro * value_size]
        """
        batch_size, seq_len, _ = sequence_input.shape
        query = self.trans_seq(sequence_input).transpose(2, 1).reshape(batch_size * self.n_pro, seq_len, self.key_size)
        curkey = self.trans_cur(current_input).reshape(batch_size * self.n_pro, self.key_size, 1)
        # b * n * seq_len
        dot = torch.softmax(torch.bmm(query, curkey).squeeze(1) / self.sqrt_k, 1)
        if len(dot.shape) == 2:
            dot = dot.unsqueeze(-1)
        value = dot.expand(batch_size * self.n_pro, seq_len, self.value_size)
        skip_connect = value.view(batch_size, self.n_pro, seq_len, -1).transpose(2, 1).reshape(batch_size, seq_len, -1)
        res = self.layer_norm(skip_connect)
        return res

class Current_variables_block(nn.Module):
    def __init__(self, static_variables: int, continuous_variables: int, embedding_dim):
        """
        :param static_variables: the number of categories characters, type: int ,which can be convert into long-tensor
        :param continuous_variables: the number of continuous characters, type: float
        :param embedding_dim:
        """
        super(Current_variables_block, self).__init__()
        self.static_variables = static_variables
        self.continuous_variables = continuous_variables
        self.embedding_dim = embedding_dim
        if static_variables > 0:
            self.static_embedding = Static_variables_embedding(self.static_variables, embedding_dim)
        if continuous_variables > 0:
            self.continuous_embedding = nn.Linear(self.continuous_variables, self.embedding_dim * self.continuous_variables)

    def forward(self, static_input, continuous_input):
        current_embedding = None
        if self.static_variables > 0:
            static_embedding = self.static_embedding(static_input).view(-1, self.static_variables * self.embedding_dim)
            current_embedding = static_embedding
        if self.continuous_variables > 0:
            continuous_embedding = self.continuous_embedding(continuous_input)
            if self.static_variables > 0:
                current_embedding = torch.cat([static_embedding, continuous_embedding], dim=-1)
            else:
                current_embedding = continuous_embedding
        return current_embedding


class Static_variables_embedding(nn.Module):
    def __init__(self, static_variables: int, embedding_dim: int):
        """
        :param static_variables: the number of categories characters, type: int ,which can be convert into long-tensor
        :param embedding_dim:
        """
        super(Static_variables_embedding, self).__init__()
        self.static_variables = static_variables
        self.embedding_dim = embedding_dim

        self.static_embedding = nn.Embedding(self.static_variables, self.embedding_dim)

    def forward(self, static_input):
        """
        :param static_input: long-type tensor, shaped [ bs, num_characters]
        :return: embedding-tensor, shaped [ bs, num_characters * embedding_dim]
        """
        return self.static_embedding(static_input).view(-1, self.static_variables * self.embedding_dim)
