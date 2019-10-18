import os
from abc import ABC

from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Callable, Union, List, Tuple

import data
from disentangle.dataset_generator import InfiniteDataset, FrankensteinDataset

from disentangle.layers import get_fc_net, GradientReversal, GradientReversalFunction, get_1by1_conv1d_net, IdentityLayer, get_block_fc_net, \
    L2Norm, BlockLinear
from utils import init

from sklearn import metrics
import numpy as np
from utils import NP
import torch.nn.functional as NN

from sklearn.decomposition import PCA
from tsnecuda import TSNE
from torch.nn import functional as F

def set_grad(parameters_or_module, value: bool):
    parameters = parameters_or_module.parameters() if isinstance(parameters_or_module, nn.Module) else parameters_or_module
    for param in parameters:
        param.requires_grad = value

class LossBox:
    def __init__(self, str_format='4.5f'):
        self._d = {}
        self._str_format = str_format

    def append(self, name, loss):
        if name not in self._d.keys():
            self._d[name] = []
        self._d[name].append(loss.detach().cpu())

    def __str__(self):
        s = ""
        for key in self._d.keys():
            s += f"  - Loss {key}:  {torch.stack(self._d[key]).mean():{self._str_format}}\n"
        return s


class Model(nn.Module, ABC):
    def __init__(self):
        super(Model, self).__init__()
        self._device = None

    @property
    def device(self): return self._device

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self._device = device
        return self


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, nb_attributes):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim * nb_attributes)
        self._latent_dim = latent_dim
        self._nb_latents = nb_attributes

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        return self.mu(hidden)


class AttrDecoder(nn.Module):
    def __init__(self, latent_dim, nb_attributes, hidden_dim, output_dim=1):
        super().__init__()
        self.latent_to_hidden = BlockLinear(latent_dim, hidden_dim, nb_attributes)
        self.hidden_to_out = BlockLinear(hidden_dim, output_dim, nb_attributes)
        # self.latent_to_hidden = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        # self.hidden_to_out = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        #return torch.sigmoid(self.latent_to_hidden(x))
        x = torch.relu(self.latent_to_hidden(x))
        return torch.sigmoid(self.hidden_to_out(x))


import torch.distributions as tdist

class ADisVAE(Model):
    def __init__(self, input_dim, hidden_dim, latent_dim, nb_attributes):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._nb_attributes = nb_attributes
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, nb_attributes)
        self.attr_decoder = AttrDecoder(latent_dim, nb_attributes, int(latent_dim/2),  1)

    def forward(self, x, a_mask=None):
        z_mu = self.encoder(x)
        a1 = self.attr_decoder(z_mu)
        return a1, z_mu



    def fit_loader(self, train_loader, optimizer, alpha=(1, 1, 4)):
        self.train()
        L = LossBox()
        for i, (x, a_mask, a) in enumerate(train_loader):
            x = x.to(self.device)
            a = a.to(self.device)
            a_mask = a_mask.to(self.device)
            optimizer.zero_grad()
            a1, z_mu = self.forward(x, a_mask)
            RCLa = F.mse_loss(a1, a)  # reconstruction loss
            RCLa.backward()
            optimizer.step()
            L.append('RCL-A', RCLa);
        return L



    def idx2onehot(self, idx):
        assert idx.shape[1] == 1
        assert torch.max(idx).item() < self._n_classes
        onehot = torch.zeros(idx.size(0), self._n_classes)
        onehot.scatter_(1, idx.data, 1)
        return onehot

    def predict_tensor(self, X, bs=128, pin_memory=True, num_workers=0):
        return self.predict_loader(DataLoader(TensorDataset(X), batch_size=bs, shuffle=False, pin_memory=pin_memory,num_workers=num_workers))

    def predict_loader(self, loader):
        self.eval()
        X, Y, ZMU, ZVAR = [], [], [], []
        for i, (x,) in enumerate(loader):
            x = x.to(self.device);
            x1, a1, z_mu, z_var = self.forward(x)
            X.append(x1.detach().cpu()); Y.append(a1.detach().cpu())
            ZMU.append(z_mu.detach().cpu()); ZVAR.append(z_var.detach().cpu())
        return (torch.cat(T, dim=0) for T in (X, Y, ZMU, ZVAR))


#%% INIT

def init_exp(gpu, seed, use_valid=False, download_data=False, preprocess_data=False, dataset='AWA2'):
    device = init(gpu_index=gpu, seed=seed)
    print('\n\n\n' + ('*' * 80) + f'\n   Starting new exp with seed {seed} on device {device}\n' + ('*' * 80) + '\n')
    if download_data:
        data.download_data()
    if preprocess_data:
        data.preprocess_dataset(dataset)
    train, val, test_unseen, test_seen = data.get_dataset(dataset, use_valid=use_valid, gzsl=False, mean_sub=False, std_norm=False,
                                                          l2_norm=False)
    train, val, test_unseen, test_seen = data.normalize_dataset(train, val, test_unseen, test_seen, keys=('class_attr', 'attr'),
                                                                feats_range=(0, 1))

    # attr_mean = np.mean(train['attr'], axis=0)
    # attr_std = np.std(train['attr'], axis=0)
    # for s in  [train, val, test_unseen, test_seen]:
    #     if s is not None:
    #         s['attr'] -= attr_mean
    #
    #         labels = sorted(set(s['labels']))
    #         A = s['class_attr']
    #         for lidx, l in enumerate(labels):
    #             idx = np.where(s['labels'] == l)[0]
    #             A[lidx] == np.mean(s['attr'][idx], axis=0)
    #         s['class_attr'] = A

    return device, train, test_unseen, test_seen, val



# %%
from torch import optim


if __name__ == '__main__':
    nb_epochs=300
    device, train, test_unseen, test_seen, val = init_exp(gpu=1, seed=42, dataset='AWA2')
    feats_dim = len(train['feats'][0])
    nb_train_classes, nb_attributes = train['class_attr'].shape
    nb_test_classes, _ = test_unseen['class_attr'].shape

    vae = ADisVAE(feats_dim, 1500, 32, nb_attributes).to(device)
    trainset_w_cls = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels'])[:,None].long())
    trainset = TensorDataset(torch.tensor(train['feats']).float(),
                             torch.tensor(train['attr_bin']).float(), # mask
                             torch.tensor(train['attr']).float() ) # attr-reconstruction


    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)
    #optimizer = optim.Adam(vae.parameters(), lr=.0008)
    optimizer = optim.Adam(vae.parameters(), lr=.001)

    for ep in range(nb_epochs):

        lossbox = vae.fit_loader(train_loader, optimizer, alpha=(0, 100,.1)) # RCLx + RCLa + KLD
        #lossbox = vae.fit_loader(train_loader, optimizer, alpha=(.001,50,2))
        print(f'====> Epoch: {ep+1}')
        print(lossbox)

        if (ep+1)%2222 == 0:
            # Test on unseen-test-set:
            classifier = nn.Sequential(#nn.Linear(feats_dim, feats_dim),
                                       #L2Norm(10., norm_while_test=True),
                                       nn.Linear(feats_dim, nb_test_classes))

            zs_test(vae, classifier, train, test_unseen,
                    nb_gen_class_samples=200, adapt_epochs=8, adapt_lr=.001, adapt_bs=128, attrs_key='class_attr_bin', device=device,
                    plot_tsne=False, plot_tsne_l2_norm=True)

            # # Test on seen-train-set:
            # classifier = get_fc_net(feats_dim, hidden_sizes=None, output_size=nb_train_classes)
            # zs_test(vae, classifier, train, train,
            #         nb_gen_class_samples=200, adapt_epochs=5, adapt_lr=.001, adapt_bs=128, attrs_key='class_attr_bin', device=device,
            #         plot_tsne=True)



            # vae.tsne_plot([trainset_w_cls.tensors, (rec_x, rec_y)], append_title=f" - Epoch={ep + 1}",
            #               perplexity = 50, num_neighbors = 6, early_exaggeration = 12)

    # TODO:

#%% TRASHCAN



#%%

