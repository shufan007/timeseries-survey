import torch, math
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


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """[summary]
        :param d_model: [description]
        :type d_model: int
        :param max_len: [description], defaults to 5000
        :type max_len: int, optional
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """[summary]

        :param x: [description]
        :type x: [type]
        :return: [description]
        :rtype: [type]
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        """Create the token embedding

        :param c_in: [description]
        :type c_in: [type]
        :param d_model: [description]
        :type d_model: [type]
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create the token embedding

        :param x: The tensor passed to create the token embedding with shape (batch_size, seq_len, feature_in)
        :type x: torch.Tensor
        :return: [description]
        :rtype: torch.Tensor
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in: torch.Tensor, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', lowest_level=3):
        """A class to create

        :param d_model: The model embedding dimension
        :type d_model: int
        :param embed_tsype: [description], defaults to 'fixed'
        :type embed_type: str, optional
        :param lowest_level: [description], defaults to 4
        :type lowest_level: int, optional
        """
        super(TemporalEmbedding, self).__init__()
        minute_size = 6
        hour_size = 24
        day_size = 245
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        lowest_level_map = { "day_embed": Embed(day_size, d_model),
                             "hour_embed": Embed(hour_size, d_model),
                            "minute_embed": Embed(minute_size, d_model)}
        for i in range(0, lowest_level):
            setattr(self, list(lowest_level_map.keys())[i], list(lowest_level_map.values())[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates the datetime embedding component
        :param x: A PyTorch tensor of shape (batch_size, seq_len, n_feats).
        n_feats is formatted in the following manner.
        following way ()
        :type x: torch.Tensor
        :return: The datetime embedding of shape (batch_size, seq_len, 1)
        :rtype: torch.Tensor
        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 2]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 1]) if hasattr(self, 'hour_embed') else 0.
        day_x = self.day_embed(x[:, :, 0])
        return hour_x  + day_x + minute_x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)



# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/100_models.layers.ipynb (unless otherwise specified).

# Cell
"""
from fastai.layers import *
from fastai.losses import *
from ..imports import *
from ..utils import *
"""

from torch import Tensor
from collections import defaultdict
from typing import Optional, Callable, Any, Tuple


def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


class Flatten(nn.Module):
    def __init__(self, full=False):
        super(Flatten, self).__init__()
        self.full = full

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        "Flatten `x` to a single dimension, e.g. at end of a model. `full` for rank-1 tensor"
        return x.view(-1) if self.full else x.view(x.size(0), -1)


# Cell
class SmeLU(nn.Module):
    "Smooth ReLU activation function based on https://arxiv.org/pdf/2202.06499.pdf"

    def __init__(self,
        beta: float = 2. # Beta value
        ) -> None:
        super().__init__()
        self.beta = abs(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.abs(x) <= self.beta, ((x + self.beta) ** 2) / (4. * self.beta), F.relu(x))

# Cell
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class ReGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.relu(gates)

# Cell
pytorch_acts = [nn.ELU, nn.LeakyReLU, nn.PReLU, nn.ReLU, nn.ReLU6, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid, nn.Softplus,
nn.Tanh, nn.Softmax, GEGLU, ReGLU]
pytorch_act_names = [a.__name__.lower() for a in pytorch_acts]

def get_act_fn(act, **act_kwargs):
    if act is None: return
    elif isinstance(act, nn.Module): return act
    elif callable(act): return act(**act_kwargs)
    idx = pytorch_act_names.index(act.lower())
    return pytorch_acts[idx](**act_kwargs)

# Cell
class TensorBase(Tensor):
    "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
    debug,_opt = False,defaultdict(list)
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res

    @classmethod
    def _before_cast(cls, x): return tensor(x)
    def __repr__(self): return re.sub('tensor', self.__class__.__name__, super().__repr__())

    def __reduce_ex__(self,proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
        args = args + (self.requires_grad, OrderedDict())
        f = torch._utils._rebuild_qtensor if self.is_quantized else  torch._utils._rebuild_tensor_v2
        return (_rebuild_from_type, (f, type(self), args, self.__dict__))

    @classmethod
    def register_func(cls, func, *oks): cls._opt[func].append(oks)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
        if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
        res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
        dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
        if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)
        return res

    def new_tensor(self, size, dtype=None, device=None, requires_grad=False):
        cls = type(self)
        return self.as_subclass(Tensor).new_tensor(size, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

    def new_ones(self, data, dtype=None, device=None, requires_grad=False):
        cls = type(self)
        return self.as_subclass(Tensor).new_ones(data, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

    def new(self, x=None):
        cls = type(self)
        res = self.as_subclass(Tensor).new() if x is None else self.as_subclass(Tensor).new(x)
        return res.as_subclass(cls)

    def requires_grad_(self, requires_grad=True):
        # Workaround https://github.com/pytorch/pytorch/issues/50219
        self.requires_grad = requires_grad
        return self

# Cell
class NumpyTensor(TensorBase):
    "Returns a `tensor` with subclass `NumpyTensor` that has a show method"

    def __new__(cls, o, dtype=None, device=None, **kwargs):
        if dtype is not None or device is not None: o = torch.as_tensor(o, dtype=dtype, device=device)
        res = cast(o, cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res

    @property
    def data(self): return cast(self, Tensor)

    def __repr__(self):
        if self.ndim > 0: return f'NumpyTensor(shape:{tuple(self.shape)}, device={self.device}, dtype={self.dtype})'
        else: return f'NumpyTensor([{self}], device={self.device}, dtype={self.dtype})'


class TSMaskTensor(NumpyTensor):
    def __repr__(self):
        if self.ndim == 0: return f'{self}'
        else: return f'TSMaskTensor(shape:{tuple(self.shape)}, device={self.device}, dtype={self.dtype})'


# Cell
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

# Cell
def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

# Cell
def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


# Cell
def same_padding1d(seq_len, ks, stride=1, dilation=1):
    "Same padding formula as used in Tensorflow"
    p = (seq_len - 1) * stride + (ks - 1) * dilation + 1 - seq_len
    return p // 2, p - p // 2


class Pad1d(nn.ConstantPad1d):
    def __init__(self, padding, value=0.):
        super().__init__(padding, value)


#@delegates(nn.Conv1d)
class SameConv1d(nn.Module):
    "Conv1d with padding='same'"
    def __init__(self, ni, nf, ks=3, stride=1, dilation=1, **kwargs):
        super(SameConv1d, self).__init__()
        self.ks, self.stride, self.dilation = ks, stride, dilation
        self.conv1d_same = nn.Conv1d(ni, nf, ks, stride=stride, dilation=dilation, **kwargs)
        self.weight = self.conv1d_same.weight
        self.bias = self.conv1d_same.bias
        self.pad = Pad1d

    def forward(self, x):
        self.padding = same_padding1d(x.shape[-1], self.ks, dilation=self.dilation) #stride=self.stride not used in padding calculation!
        return self.conv1d_same(self.pad(self.padding)(x))


# Cell
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, ni, nf, ks, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__(ni, nf, kernel_size=ks, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        self.__padding = (ks - 1) * dilation
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

# Cell
#@delegates(nn.Conv1d)
def Conv1d(ni, nf, kernel_size=None, ks=None, stride=1, padding='same', dilation=1, init='auto', bias_std=0.01, **kwargs):
    "conv1d layer with padding='same', 'causal', 'valid', or any integer (defaults to 'same')"
    assert not (kernel_size and ks), 'use kernel_size or ks but not both simultaneously'
    assert kernel_size is not None or ks is not None, 'you need to pass a ks'
    kernel_size = kernel_size or ks
    if padding == 'same':
        if kernel_size%2==1:
            conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=kernel_size//2 * dilation, dilation=dilation, **kwargs)
        else:
            conv = SameConv1d(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    elif padding == 'causal': conv = CausalConv1d(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    elif padding == 'valid': conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)
    else: conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=padding, dilation=dilation, **kwargs)
    init_linear(conv, None, init=init, bias_std=bias_std)
    return conv


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super(Transpose, self).__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
    def __repr__(self):
        if self.contiguous: return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])}).contiguous()"
        else: return f"{self.__class__.__name__}({', '.join([str(d) for d in self.dims])})"


# Cell
class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super(ScaledDotProductAttention, self).__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]

        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

# Cell
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super(MultiheadAttention, self).__init__()
        """Multi Head Attention Layer

        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """

        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


