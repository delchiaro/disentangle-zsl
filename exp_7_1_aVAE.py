import os
from abc import ABC

from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Callable, Union, List, Set, Tuple

import data
from disentangle.dataset_generator import InfiniteDataset, FrankensteinDataset

from disentangle.layers import get_fc_net, GradientReversal, GradientReversalFunction, get_1by1_conv1d_net, IdentityLayer, get_block_fc_net
from utils import init, to_categorical

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
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim + n_classes, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        return self.mu(hidden), self.var(hidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.latent_to_hidden(x))
        return self.hidden_to_out(x)

class CVAE(Model):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_attributes):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._n_attributes = n_attributes
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_attributes)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_attributes)

    def forward(self, x, a):
        x = torch.cat((x, a), dim=1)
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        z = torch.cat((x_sample, a), dim=1)

        # decode
        generated_x = self.decoder(z)
        return generated_x, z_mu, z_var

    def compute_loss(self, x, reconstructed_x, mean, log_var):
        RCL = F.l1_loss(reconstructed_x, x, size_average=False)   # reconstruction loss
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  # kl divergence loss
        return RCL + KLD

    def idx2onehot(self, idx):
        assert idx.shape[1] == 1
        assert torch.max(idx).item() < self._n_attributes
        onehot = torch.zeros(idx.size(0), self._n_attributes)
        onehot.scatter_(1, idx.data, 1)
        return onehot

    def fit_loader(self, train_loader, optimizer):
        self.train()
        train_loss = 0
        for i, (x, a) in enumerate(train_loader):
            x = x.to(self.device)
            #y = self.idx2onehot(y.view(-1, 1)) # convert y into one-hot encoding
            a = a.to(self.device)

            optimizer.zero_grad()
            reconstructed_x, z_mu, z_var = self(x, a)
            loss = self.compute_loss(x, reconstructed_x, z_mu, z_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        return train_loss

    def generate(self, nb_gen, attributes=None):
        # create a random latent vector
        if attributes is None:
            attributes = torch.arange(0, self._n_attributes).long()
            attributes = self.idx2onehot(attributes[:, None])
        elif isinstance(attributes, list) or isinstance(attributes, set) or isinstance(attributes, tuple):
            attributes = torch.tensor(attributes).float()
        elif isinstance(attributes, int):
            attributes = torch.tensor([attributes]).float()
        attributes = attributes.repeat_interleave(nb_gen,dim=0)
        self.eval()
        with torch.no_grad():
            z = torch.randn(len(attributes), self._latent_dim).to(self.device)
            #y = self.idx2onehot(attributes[:, None]).to(self.device, dtype=z.dtype)
            y=attributes.to(self.device, dtype=z.dtype)
            z = torch.cat((z, y), dim=1)
            x_gen = self.decoder(z)
        return x_gen, attributes



    def tsne_plot(self, data: Tuple[torch.FloatTensor], markers=['o', 'X', '.'], alphas=[1, 0.5, 0.3],
                  nb_pca_components=None, append_title='', legend=False, savepath=None, figsize=(14, 8),  dpi=250,
                  **kwargs):
        if isinstance(data[0], torch.Tensor):
            data = [data]
        Xs = [NP(d[0]) for d in data]
        X = np.concatenate(Xs, axis=0)

        As = [NP(d[1]) for d in data]
        A = np.concatenate(As, axis=0)
        try:
            Ys = [NP(d[2]) for d in data]
            Y =  np.concatenate(Ys, axis=0)
            compute_Y = False
        except:
            Ys = None
            Y =  None
            compute_Y = True


        unique_A = np.unique(A, axis=0)

        ### PCA
        if nb_pca_components is not None:
            pca = PCA(n_components=nb_pca_components)
            print("Fitting PCA and transforming...")
            pca.fit(X)
            X = pca.transform(X)

        ### TSNE FIT
        print("Fitting TSNE and transforming...")
        tsne_data = TSNE(n_components=2, **kwargs).fit_transform(X)

        if compute_Y:
            Y = np.zeros([len(A)], dtype='long') - 1
            for y, a in enumerate(unique_A):
                idx = np.argwhere(np.all(A == a, axis=1)).squeeze(axis=1)
                Y[idx] = y

        b = len(Xs[0])
        Ts = [tsne_data[:b]]

        if compute_Y:
            Ys = [Y[:b]]
        for x in (Xs[1:]):
            new_b = b+len(x)
            Ts.append(tsne_data[b:new_b])
            if compute_Y:
                Ys.append(Y[b:new_b])
            b = new_b

        # PLOT TSNE
        from matplotlib import pyplot as plt
        import seaborn as sns
        fig = plt.figure(figsize=figsize, dpi=dpi)
        title = f'tsne {append_title}'
        fig.suptitle(title, fontsize=16)
        for T, Y, m, a in zip(Ts, Ys, markers[:len(Ts)], alphas[:len(Ts)]):
            sns.scatterplot(x=T[:, 0], y=T[:, 1], hue=Y, marker=m, alpha=a, palette=sns.color_palette("hls", len(set(Y))), legend=legend)

        if savepath is not None:
            os.makedirs(savepath, exist_ok=True)
            plt.savefig(os.path.join(savepath, title + '.png'))
        else:
            plt.show()

#%% EXP

def init_exp(gpu, seed, use_valid=False, download_data=False, preprocess_data=False, dataset='AWA2'):
    device = init(gpu_index=gpu, seed=seed)
    print('\n\n\n' + ('*' * 80) + f'\n   Starting new exp with seed {seed} on device {device}\n' + ('*' * 80) + '\n')
    if download_data:
        data.download_data()
    if preprocess_data:
        data.preprocess_dataset(dataset)
    train, val, test_unseen, test_seen = data.get_dataset(dataset, use_valid=use_valid, gzsl=False, mean_sub=False, std_norm=False,
                                                          l2_norm=False)
    train, val, test_unseen, test_seen = data.normalize_dataset(train, val, test_unseen, test_seen, keys=('class_attr',),
                                                                feats_range=(0, 1))
    return device, train, test_unseen, test_seen, val


#%% TEST

# %%
from torch import optim


if __name__ == '__main__':
    nb_epochs=300
    device, train, test_unseen, test_seen, val = init_exp(gpu=0, seed=42, dataset='AWA2')
    feats_dim = len(train['feats'][0])
    nb_train_classes, nb_attributes = train['class_attr'].shape
    #nb_test_classes, _ = test_unseen['class_attr'].shape

    cvae = CVAE(feats_dim, 1500, 768, nb_attributes).to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=.001)
    #A = torch.tensor(train['class_attr'])
    cls_trainset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['attr_bin']).float(), torch.tensor(train['labels']).long())
    attr_trainset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['attr_bin']).float())
    train_loader = DataLoader(attr_trainset, batch_size=128, shuffle=True, drop_last=True)
    A = torch.tensor(train['class_attr_bin']).to(device)
    for ep in range(nb_epochs):
        loss = cvae.fit_loader(train_loader, optimizer)
        print(f'====> Epoch: {ep} Average loss: {loss/len(train_loader.dataset):.4f}')
        if (ep+1)%20 == 0:
            gen_data = cvae.generate(nb_gen=200, attributes=A)
            #cvae.tsne_plot([cls_trainset.tensors], append_title=f" - Epoch={ep+1}", perplexity=90)
            cvae.tsne_plot([attr_trainset.tensors, gen_data], append_title=f" - Epoch={ep+1}",
                           perplexity=50, num_neighbors=6, early_exaggeration=12
                           )




#%% TRASHCAN
