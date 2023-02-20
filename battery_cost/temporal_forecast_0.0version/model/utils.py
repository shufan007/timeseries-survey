import torch
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self, channel):
        """
        GLU (gated linear unit) application of temporal prediction

        activation parameters:
            dim_in: the number of features
            dim_out: the number of features outputted

        math:
            x = f_1 ( x ) * sigmoid (f_2 ( x )), where f_1 and f_2 use a linear weighted function

        GLU parameters:
            channels: the input and output size of input

        """

        super(GLU, self).__init__()
        self.dim_in = channel
        self.activation = nn.Linear(self.dim_in, 2 * self.dim_in)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        :param input:
            tensor: [..., F]
        :return:
            tensor: [..., F]
        """
        _P_Q = self.activation(input)
        _P, _Q = torch.split(_P_Q, self.dim_in, dim=-1)
        return _P * self.sigmoid(_Q)


class fully_connected_layer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation_type='relu', norm_layer=False, last_dropout=0):
        """
        Initialize fullyConnectedNet.
        Parameters
        ----------
        input_size – The number of expected features in the input x  -> scalar
        hidden_size – The numbers of features in the hidden layer h  -> list
        output_size  – The number of expected features in the output x  -> scalar
        input -> (batch, in_features)
        :return
        output -> (batch, out_features)
        """
        super(fully_connected_layer, self).__init__()

        self.input_size = input_size
        # list
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.norm_layer = norm_layer
        # batch normal
        if norm_layer:
            norm_layer_List = []

        #activation type:
        if activation_type == 'relu':
            activation_func = nn.ReLU()
        elif activation_type == 'tanh':
            activation_func = nn.Tanh()
        elif activation_type == 'sigmoid':
            activation_func = nn.Sigmoid()
        else:
            raise ValueError("we only support with [relu, tanh, sigmoid], please meke sure the right name or add this function into fully_connected_layer")

        fc_List = []
        activation_List = []

        for index in range(len(self.hidden_size)):
            if index != 0:
                input_size = self.hidden_size[index - 1]
            # fc
            fc = nn.Linear(input_size, self.hidden_size[index])
            setattr(self, f'fully_connected_{index}', fc)
            fc_List.append(fc)
            # batch normal
            if norm_layer:
                bn = nn.LayerNorm(self.hidden_size[index])
                setattr(self, f'LayerNorm_{index}', bn)
                norm_layer_List.append(bn)
            # activation
            setattr(self, f'{activation_type, index}', activation_func)
            activation_List.append(activation_func)
        # last out update
        self.last_fc = nn.Linear(self.hidden_size[-1], self.output_size)
        self.last_dropout = nn.Dropout(p=last_dropout)

        self.fc_List = nn.ModuleList(fc_List)
        if norm_layer:
            self.norm_layer_List = nn.ModuleList(norm_layer_List)
        self.activation_List = nn.ModuleList(activation_List)

    def forward(self, input_tensor):
        """
        :param input_tensor:
            2-D Tensor  (..., input_size)
        :return:
            2-D Tensor (..., output_size)
            output_tensor
        """
        for idx in range(len(self.fc_List)):
            out = self.fc_List[idx](input_tensor)
            if self.norm_layer:
                out = self.norm_layer_List[idx](out)
            out = self.activation_List[idx](out)
            input_tensor = out
        # (batch, output_size)

        output_tensor = self.last_fc(self.last_dropout(input_tensor))

        return output_tensor
