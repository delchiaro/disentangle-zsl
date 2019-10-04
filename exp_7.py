import os
from abc import ABC

from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Callable, Union, List

import data
from disentangle.dataset_generator import InfiniteDataset, FrankensteinDataset

from disentangle.layers import get_fc_net, GradientReversal, GradientReversalFunction, get_1by1_conv1d_net, IdentityLayer, get_block_fc_net
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
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._n_classes = n_classes
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_classes)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)
        return generated_x, z_mu, z_var

    def compute_loss(self, x, reconstructed_x, mean, log_var):
        RCL = F.l1_loss(reconstructed_x, x, size_average=False)   # reconstruction loss
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  # kl divergence loss
        return RCL + KLD

    def idx2onehot(self, idx):
        assert idx.shape[1] == 1
        assert torch.max(idx).item() < self._n_classes
        onehot = torch.zeros(idx.size(0), self._n_classes)
        onehot.scatter_(1, idx.data, 1)
        return onehot

    def fit_loader(self, train_loader, optimizer):
        self.train()
        train_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = self.idx2onehot(y.view(-1, 1)) # convert y into one-hot encoding
            y = y.to(self.device)

            optimizer.zero_grad()
            reconstructed_x, z_mu, z_var = self(x, y)
            loss = self.compute_loss(x, reconstructed_x, z_mu, z_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        return train_loss

    def generate(self, nb_gen, label=None):
        # create a random latent vector
        if label is None:
            label = torch.arange(0, self._n_classes).long()
        elif isinstance(label, list) or isinstance(label, set) or isinstance(label, tuple):
            label = torch.tensor(label).long()
        elif isinstance(label, int):
            label = torch.tensor([label]).long()
        label = label.repeat_interleave(nb_gen)
        self.eval()
        with torch.no_grad():
            z = torch.randn(len(label), self._latent_dim).to(self.device)
            y = self.idx2onehot(label[:, None]).to(self.device, dtype=z.dtype)
            z = torch.cat((z, y), dim=1)
            x_gen = self.decoder(z)
        return x_gen, label

    def tsne_plot(self, real_data: TensorDataset=None, gen_data: TensorDataset=None, nb_gen=None,
                  nb_pca_components=None, append_title='', legend=False, savepath=None, dpi=250):
        if gen_data is None:
            Xg, Yg = (NP(T) for T in self.generate(nb_gen))
        else:
            Xg, Yg =  (NP(T) for T in gen_data.tensors)

        if real_data is None:
            X, Y = None
            all_classes = np.unique(Yg)
        else:
            X, Y = (NP(T) for T in real_data.tensors) if real_data is not None else (None, None)
            all_classes = np.unique(np.concatenate([Y, Yg], axis=0))

        nb_classes = len(all_classes)

        ### PCA
        if nb_pca_components is not None:
            pca = PCA(n_components=nb_pca_components)
            print("Fitting PCA and transforming...")
            pca.fit(X)
            X = pca.transform(X)
            Xg = pca.transform(Xg)

        ### TSNE FIT
        print("Fitting TSNE and transforming...")
        feats = np.concatenate([X, Xg], axis=0) if X is not None else Xg
        embeddings_gen = TSNE(n_components=2).fit_transform(feats)
        if X is not None:
            embeddings_real = embeddings_gen[:len(X)]
            embeddings_gen = embeddings_gen[len(X):]

        # PLOT TSNE
        from matplotlib import pyplot as plt
        import seaborn as sns
        fig = plt.figure(figsize=(18, 8), dpi=dpi)
        title = f'tsne {append_title}'
        fig.suptitle(title, fontsize=16)
        if X is not None:
            sns.scatterplot(x=embeddings_real[:, 0], y=embeddings_real[:, 1], hue=Y,
                            palette=sns.color_palette("hls", nb_classes), legend=legend, alpha=0.4)
        sns.scatterplot(x=embeddings_gen[:, 0], y=embeddings_gen[:, 1], hue=Yg,
                        palette=sns.color_palette("hls", nb_classes), legend=legend, alpha=1)

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

    cvae = CVAE(feats_dim, 1500, 768, nb_train_classes).to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=.001)
    trainset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)

    for ep in range(nb_epochs):
        loss = cvae.fit_loader(train_loader, optimizer)
        print(f'====> Epoch: {ep} Average loss: {loss/len(train_loader.dataset):.4f}')
        if (ep+1)%50 == 0:
            cvae.tsne_plot(trainset, nb_gen=30, append_title=f" - Epoch={ep+1}")



#%% TRASHCAN
