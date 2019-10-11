import math

import torch
from torch.autograd import Function
from torch import nn
from torch.nn import init


class IdentityLayer(nn.Module):
    def forward(self, *x):
        if len(x) == 1:
            return x[0]
        else:
            return x

def get_fc_net(input_size: int, hidden_sizes: tuple, output_size: int = None, hidden_activations=nn.LeakyReLU(), out_activation=None, device=None):
    hidden_sizes = [] if hidden_sizes is None else hidden_sizes
    hidden_sizes = list(hidden_sizes) + ([output_size] if output_size is not None else [])
    if len(hidden_sizes) > 0:
        layers = [nn.Linear(input_size, hidden_sizes[0])]
        prev_size = hidden_sizes[0]
        for size in hidden_sizes[1:]:
            if hidden_activations is not None:
                layers.append(hidden_activations)
            layers.append(nn.Linear(prev_size, size))
            prev_size = size
        if out_activation is not None:
            layers.append(out_activation)
        net = nn.Sequential(*layers)
        out_dim = hidden_sizes[-1]
    else:
        net = nn.Identity()
        out_dim = input_size
    net.out_dim = out_dim
    return net.to(device)

def get_1by1_conv1d_net(in_channels: int, hidden_channels: tuple, output_channels: int = None, hidden_activations=nn.LeakyReLU(),
                        out_activation=None):
    hidden_channels = [] if hidden_channels is None else hidden_channels
    hidden_channels = list(hidden_channels) + ([output_channels] if output_channels is not None else [])
    if len(hidden_channels) > 0:
        layers = []
        prev_channels = in_channels
        for channels in hidden_channels[:-1]:
            layers.append(nn.Conv1d(prev_channels, out_channels=channels, kernel_size=1, stride=1))
            prev_channels = channels
            if hidden_activations is not None:
                layers.append(hidden_activations)
        layers.append(nn.Conv1d(prev_channels, out_channels=hidden_channels[-1], kernel_size=1, stride=1))
        if out_activation is not None:
            layers.append(out_activation)
        net = nn.Sequential(*layers)
        net.out_channels = hidden_channels[-1]
    else:
        net = nn.Identity()
        net.out_channels = in_channels
    return net

def get_block_fc_net(n_blocks:int, input_size: int, hidden_sizes: tuple, output_size: int = None,
                     hidden_activations=nn.LeakyReLU(), out_activation=None, device=None):
    hidden_sizes = [] if hidden_sizes is None else hidden_sizes
    hidden_sizes = list(hidden_sizes) + ([output_size] if output_size is not None else [])
    if len(hidden_sizes) > 0:
        layers = [BlockLinear(input_size, hidden_sizes[0], n_blocks)]
        prev_size = hidden_sizes[0]
        for size in hidden_sizes[1:]:
            if hidden_activations is not None:
                layers.append(hidden_activations)
            layers.append(BlockLinear(prev_size, size, n_blocks))
            prev_size = size
        if out_activation is not None:
            layers.append(out_activation)
        net = nn.Sequential(*layers)
        out_dim = hidden_sizes[-1]
    else:
        net = nn.Identity()
        out_dim = input_size
    net.out_dim = out_dim
    return net.to(device)


# From: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)



















import torch
from torch.autograd import Function


class MaskedGradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, mask, lambda_):
        ctx.mask = mask
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_outputs):
        lambda_ = ctx.lambda_
        lambda_ = grad_outputs.new_tensor(lambda_)
        #pdb.set_trace()
        dx = ctx.mask.unsqueeze(1).repeat([1, grad_outputs.shape[1]]) * lambda_ * grad_outputs
        #print(dx)
        return dx, None, None


class MaskedGradientReversal(torch.nn.Module):
    def __init__(self, mask, lambda_):
        super(MaskedGradientReversal, self).__init__()
        self.mask_ = mask
        self.lambda_ = lambda_

    def forward(self, x):
        return MaskedGradientReversalFunction.apply(x[0], self.mask_, self.lambda_)











import torch
def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1).to(m.device)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


#%%

import torch.nn.functional as F
class BlockLinear(nn.Module):
    r"""Applies a block-linear transformation to the incoming data:
    :math:`\\x = [x_1, x_2, ..., x_{n\_blocks}]\\`
    :math:`y_k = x_k  A_{k} ^{T} + b_k\\`
    where the weights A of the linear transformation is a diagonal block matrix with
    n_blocks blocks each of dimension out_block_dim*in_block_dim and zeros
    outside of the blocks.
    This layer is equivalent to n_blocks linear layers each of which is applied to
    n_blocks different inputs with dim in_block_dim, that are concatenated in the
    actual input x.

    Args:
        in_block_dim, out_block_dim, n_blocks, bias=True

        in_block_dim: size of each input sample
        out_block_dim: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *,  H_{in} )` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{n\_blocks}\times \text{in\_block\_dim}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out}  = \text{n\_blocks}\times \text{out\_block\_dim}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::
        >>> l = BlockLinear(2, 2, 3)
        >>> l.weight.data.zero_()
        >>> l.weight.data[0] += 1
        >>> l.weight.data[1] += 2
        >>> l.weight.data[2] += 3
        >>> l.bias.data.zero_()
        >>> t = torch.ones([1, 6])
        >>> l(t)
        # out: 2, 2, 4, 4, 6, 6

    """
    __constants__ = ['bias']

    def __init__(self, in_block_dim, out_block_dim, n_blocks, bias=True):
        super(BlockLinear, self).__init__()
        self.in_block_dim = in_block_dim
        self.out_block_dim = out_block_dim
        self.n_blocks = n_blocks

        self.weight =  nn.Parameter(data=torch.ones(n_blocks, out_block_dim, in_block_dim))
        #self.weight = block_diag(self.params)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_block_dim*n_blocks))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    @property
    def in_features(self):
        return self.n_blocks * self.in_block_dim

    @property
    def out_features(self):
        return self.n_blocks * self.out_block_dim


    def reset_parameters(self):
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, block_diag(self.weight), self.bias)

    def extra_repr(self):
        return f'in_blocK_dim={self.in_block_dim}, out_block_dim={self.out_block_dim}, ' \
               f'n_blocks={self.n_blocks} bias={self.bias is not None}'
#%%


import numpy as np
class L2Norm(nn.Module):
    def __init__(self, alpha, dim=1, norm_while_test=True, epsilon=np.finfo(float).eps):
        super().__init__()
        self.alpha = alpha
        self.dim = dim
        self.normalize_during_test=norm_while_test
        self.epsilon = epsilon

    def forward(self, x):
        if not self.training and not self.normalize_during_test:
            return x
        else:
            x = (x / (torch.norm(x, p=2, dim=self.dim)[:, None]+self.epsilon))
            return x * self.alpha

    def extra_repr(self):
        return f'alpha={self.alpha}' + (', norm-while-test=True' if self.normalize_during_test else '')

