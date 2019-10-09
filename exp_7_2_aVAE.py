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
    def __init__(self, input_dim, hidden_dim, latent_dim, n_attrs):
        super().__init__()
        self.linear = nn.Linear(input_dim + n_attrs, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        return self.mu(hidden), self.var(hidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_attrs):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim + n_attrs, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.latent_to_hidden(x))
        return self.hidden_to_out(x)


class AttrDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_attrs):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, n_attrs)

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
        self.attr_decoder = AttrDecoder(latent_dim, hidden_dim, n_attributes)

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
        generated_a = F.sigmoid(self.attr_decoder(x_sample))
        return generated_x, generated_a, z_mu, z_var

    def compute_loss(self, x, reconstructed_x, a, reconstructed_a, mean, log_var):
        RCL_x = F.l1_loss(reconstructed_x, x, size_average=False)#/x.shape[0]   # reconstruction loss
        #RCL_a = F.binary_cross_entropy(reconstructed_a, a, size_average=False)
        RCL_a = F.l1_loss(reconstructed_a, a, size_average=False)#/x.shape[0]   # reconstruction loss
        KLD = (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()))  # kl divergence loss
        #return (RCL_x + RCL_a)/2 + KLD
        return RCL_x ,RCL_a, KLD

    def idx2onehot(self, idx):
        assert idx.shape[1] == 1
        assert torch.max(idx).item() < self._n_attributes
        onehot = torch.zeros(idx.size(0), self._n_attributes)
        onehot.scatter_(1, idx.data, 1)
        return onehot

    def fit_loader(self, train_loader, optimizer):
        self.train()
        L = LossBox()
        for i, (x, a) in enumerate(train_loader):
            x = x.to(self.device)
            a = a.to(self.device)
            optimizer.zero_grad()
            x1, a1, z_mu, z_var = self(x, a)
            RCLx, RCLa, KLD = self.compute_loss(x, x1, a, a1, z_mu, z_var)
            loss = (1*RCLx + 1*RCLa) + 2*KLD
            loss.backward()
            optimizer.step()
            L.append('RCL-X', RCLx/x.shape[0]); L.append('RCL-A', RCLa/x.shape[0]); L.append('KLD', KLD); L.append('loss', loss/x.shape[0])
        return L

    def generate(self, nb_gen, attributes=None):
        # create a random latent vector
        if attributes is None:
            attributes = torch.arange(0, self._n_attributes).long()
            attributes = self.idx2onehot(attributes[:, None])
        elif isinstance(attributes, list) or isinstance(attributes, set) or isinstance(attributes, tuple):
            attributes = torch.tensor(attributes).float()
        elif isinstance(attributes, int):
            attributes = torch.tensor([attributes]).float()
        Y = torch.arange(0, len(torch.unique(attributes, dim=1)))
        attributes = attributes.repeat_interleave(nb_gen,dim=0)
        Y = Y.repeat_interleave(nb_gen, dim=0)

        self.eval()
        with torch.no_grad():
            z = torch.randn(len(attributes), self._latent_dim).to(self.device)
            #y = self.idx2onehot(attributes[:, None]).to(self.device, dtype=z.dtype)
            a=attributes.to(self.device, dtype=z.dtype)
            z = torch.cat((z, a), dim=1)
            x_gen = self.decoder(z)
        return x_gen, attributes, Y



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

def zs_test(cvae: CVAE,
            test_classifier: nn.Module,
            test_dict,
            nb_gen_class_samples=100,
            adapt_epochs: int = 5,
            adapt_lr: float = .0001,
            adapt_bs: int = 128,
            attrs_key='class_attr_bin',
            device=None):
    test_A = torch.tensor(test_dict[attrs_key]).float().to(device)
    test_feats = torch.tensor(test_dict['feats']).float()
    test_labels = torch.tensor(test_dict['labels']).long()
    test_loader =  DataLoader(TensorDataset(test_feats, test_labels), batch_size=adapt_bs, num_workers=2)

    def test_on_test():
        losses, preds, y_trues = [], [], []
        test_classifier.eval()
        for x, y in test_loader:
            x = x.to(device); y = y.to(device)
            logit = test_classifier(x)
            loss = NN.cross_entropy(logit, y)
            losses.append(loss.detach().cpu());
            preds.append(logit.argmax(dim=1));
            y_trues.append(y)
        preds = torch.cat(preds); y_trues = torch.cat(y_trues);
        unseen_loss = torch.stack(losses).mean()
        unseen_acc = np.mean([metrics.recall_score(NP(y_trues), NP(preds), labels=[k], average=None) for k in sorted(set(NP(y_trues)))])
        return unseen_loss, unseen_acc

    ######## DATA GENERATION/PREPARATION ###########
    gen_feats, gen_attrs, gen_labels = cvae.generate(nb_gen_class_samples, test_A)
    adapt_loader = DataLoader(TensorDataset(gen_feats, gen_labels), batch_size=adapt_bs, num_workers=0, shuffle=True, )

    ######## TRAINING NEW CLASSIFIER ###########
    test_classifier.to(device)
    optim = torch.optim.Adam(test_classifier.parameters(), lr=adapt_lr)
    best_unseen_acc = 0
    best_classifier_state = None
    for ep in range(adapt_epochs):
        preds = []
        y_trues = []
        losses = []
        test_classifier.train()
        for x, y in adapt_loader:
            x = x.to(device); y = y.to(device)
            optim.zero_grad()
            logit = test_classifier(x)
            loss = NN.cross_entropy(logit, y)
            loss.backward()
            optim.step()
            losses.append(loss.detach().cpu()); preds.append(logit.argmax(dim=1)); y_trues.append(y)
        preds = torch.cat(preds); y_trues = torch.cat(y_trues); acc = (y_trues == preds).float().mean()
        classifier_losses = torch.stack(losses).mean()
        unseen_loss, unseen_acc = test_on_test()
        print(f"Classifier adaptation - Epoch {ep + 1}/{adapt_epochs}:   Loss={classifier_losses:1.5f}    Acc={acc:1.4f}  -  uLoss="
              f"{unseen_loss:1.5f}   uAcc={unseen_acc:1.5f}")
        if unseen_acc > best_unseen_acc:
            best_unseen_acc = unseen_acc
            best_classifier_state = test_classifier.state_dict()
        test_classifier.load_state_dict(best_classifier_state)
    return best_unseen_acc  # , best_classifier_state

# %%
from torch import optim


if __name__ == '__main__':
    nb_epochs=300
    device, train, test_unseen, test_seen, val = init_exp(gpu=0, seed=42, dataset='AWA2')
    feats_dim = len(train['feats'][0])
    nb_train_classes, nb_attributes = train['class_attr'].shape
    nb_test_classes, _ = test_unseen['class_attr'].shape

    cvae = CVAE(feats_dim, 1500, 768, nb_attributes).to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=.0004)

    trainset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['attr_bin']).float())
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)
    A = torch.tensor(train['class_attr_bin']).to(device)

    testset = TensorDataset(torch.tensor(test_unseen['feats']).float(), torch.tensor(test_unseen['attr_bin']).float())
    test_A = torch.tensor(test_unseen['class_attr_bin']).to(device)

    for ep in range(nb_epochs):
        lossbox = cvae.fit_loader(train_loader, optimizer)
        print(f'====> Epoch: {ep}')
        print(lossbox)

        if (ep+1)%5 == 0:
            classifier = get_fc_net(feats_dim, hidden_sizes=None, output_size=nb_test_classes)
            zs_test(cvae, classifier, test_unseen, nb_gen_class_samples=1000, adapt_epochs=10, adapt_lr=.0004, device=device)

            gen_data = cvae.generate(nb_gen=200, attributes=A)
            cvae.tsne_plot([trainset.tensors, gen_data], append_title=f"- TrainSet - Epoch={ep + 1}",
                           perplexity=50, num_neighbors=6, early_exaggeration=12)

            gen_data = cvae.generate(nb_gen=200, attributes=test_A)
            cvae.tsne_plot([testset.tensors, gen_data], append_title=f"- TestSet - Epoch={ep + 1}",
                           perplexity=50, num_neighbors=6, early_exaggeration=12)

#%% TRASHCAN
