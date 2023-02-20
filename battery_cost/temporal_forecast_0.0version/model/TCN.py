import torch.nn.functional as F
from torch import nn
from typing import *
from torch.nn import Parameter
from torch.autograd import Variable
import copy
import math
import torch
from torch.nn.utils import weight_norm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_first = True

from enum import IntEnum


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, **param):
        super(TCN, self).__init__()
        self.seq_len = int(param.get('seq_len'))
        self.horizon = int(param.get('horizon'))
        self.input_dim = int(param.get('input_dim'))
        self.output_dim = int(param.get('output_dim'))
        self.kernel_size = int(param.get('kernel_size'))
        self.hidden_dims = list(param.get('hidden_dims'))
        self.tcn = TemporalConvNet(self.input_dim, self.hidden_dims, kernel_size=self.kernel_size, dropout=0)
        self.predictor = nn.Conv1d(self.seq_len, self.horizon, kernel_size=self.hidden_dims[-1])
        # self.linear = nn.Linear(self.hidden_dims[-1], self.output_dim)

    def forward(self, input):
        """"
        Args:
            input: Required. Tensor of dimension (batch_size, seq_len, number_of_features)
        Returns:
            output: tensor of dimension (batch_size, forecast_length)
        """
        input = input[:, -self.seq_len:, -self.input_dim:]
        output = self.tcn(input.permute(0, 2, 1))  # input should have dimension (N, C, L)
        output = self.predictor(output.permute(0, 2, 1))
        output = output[:, -1, :].reshape(-1, self.output_dim)
        assert output.shape[0] == input.shape[0]
        return output