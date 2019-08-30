import torch
from torch.autograd import Function
from torch import nn

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
        net = IdentityLayer()
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
        net = IdentityLayer()
        net.out_channels = in_channels
    return net




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
