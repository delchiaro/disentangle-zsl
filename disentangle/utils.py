import torch
from torch import nn
from torch.nn import Module


def _init_weights(m: Module):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.xavier_uniform(m.bias)
def init_weights(m: Module):
    m.apply(_init_weights)



def set_grad(parameters_or_module, value: bool):
    parameters = parameters_or_module.parameters() if isinstance(parameters_or_module, nn.Module) else parameters_or_module
    for param in parameters:
        param.requires_grad = value

def interlaced_repeat(x, dim, times):
    orig_shape = x.shape
    dims = []
    for i in range(dim+1):
        dims.append(x.shape[i])
    dims.append(1)
    for i in range(dim+1, len(x.shape)):
        dims.append(x.shape[i])
    x = x.view(*dims)
    dims[dim+1]=times
    x = x.expand(dims).contiguous()
    dims = list(orig_shape)
    dims[dim] = dims[dim]*times
    x = x.view(dims)
    return x


def NP(tensor):
    return tensor.detach().cpu().numpy()


class JoinDataLoader:
    def __init__(self, master_data_loader, slave_data_loader):
        self.master = master_data_loader
        self.slave = slave_data_loader
        self.master_iter = iter(self.master)
        self.slave_iter = iter(self.slave)
        self._init_iters()

    def _init_iters(self):
        self.master_iter = iter(self.master)
        self.slave_iter = iter(self.slave)

    def __iter__(self):
        return self

    def _next_slave(self):
        try:
            return next(self.slave_iter)
        except StopIteration:
            self.slave_iter = iter(self.slave)
            return self._next_slave()

    def _next_master(self):
        try:
            return next(self.master_iter)
        except StopIteration:
            self.master_iter = iter(self.master)
            raise StopIteration

    def __next__(self):  # Python 2: def next(self)
        master_stuff = self._next_master()
        slave_stuff = self._next_slave()
        return master_stuff, slave_stuff


def join_data_loaders(master_data_loader, slave_data_loader):
    master_iter = iter(master_data_loader)
    slave_iter = iter(slave_data_loader)
    while True:
        try:
            gen_batch = next(master_iter)
        except StopIteration:
            break
        try:
            seen_batch = next(slave_iter)
        except StopIteration:
            pass
        yield gen_batch, seen_batch