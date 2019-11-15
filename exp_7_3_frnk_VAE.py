import os
from abc import ABC
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Callable, Union, List, Tuple
import data
from disentangle.dataset_generator import InfiniteDataset, FrankensteinDataset, ProbabilisticInfiniteDataset
from disentangle.layers import get_fc_net, GradientReversal, GradientReversalFunction, get_1by1_conv1d_net, IdentityLayer, get_block_fc_net, \
    L2Norm, BlockLinear
from disentangle.loss_box import LossBox
from disentangle.model import Model
from utils import init, set_seed
from sklearn import metrics
import numpy as np
from utils import NP
import torch.nn.functional as NN
from sklearn.decomposition import PCA
from tsnecuda import TSNE
from torch.nn import functional as F

from disentangle.utils import set_grad

import torch.distributions as tdist


#%% INIT
def init_exp(gpu, seed, use_valid=False, download_data=False, preprocess_data=False, dataset='AWA2', attr_in_01=True):
    device = init(gpu_index=gpu, seed=seed)
    print('\n\n\n' + ('*' * 80) + f'\n   Starting new exp with seed {seed} on device {device}\n' + ('*' * 80) + '\n')
    if download_data:
        data.download_data()
    if preprocess_data:
        data.preprocess_dataset(dataset)
    train, val, test_unseen, test_seen = data.get_dataset(dataset, use_valid=use_valid, gzsl=False, mean_sub=False, std_norm=False,
                                                          l2_norm=False)
    if attr_in_01:
        train, val, test_unseen, test_seen = data.normalize_dataset(train, val, test_unseen, test_seen, keys=('class_attr', 'attr'),
                                                                    feats_range=(0, 1))

    return device, train, test_unseen, test_seen, val

def torch_where_idx(condition):
    return condition.byte().nonzero()[:,0]

#%% NETWORKS

class Encoder(nn.Module):
    def __init__(self, nb_attributes, input_dim, hidden_dim, latent_dim, cntx_latent_dim=0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 #nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
                                 )
        self.mu = nn.Linear(hidden_dim, latent_dim * nb_attributes + cntx_latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim * nb_attributes + cntx_latent_dim)
        self._latent_dim = latent_dim
        self._cntx_latent_dim = cntx_latent_dim
        self._nb_latents = nb_attributes

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        var = self.var(h)
        return (mu[:, :self._nb_latents*self._latent_dim], mu[:, self._nb_latents*self._latent_dim:]),\
               (var[:, :self._nb_latents * self._latent_dim], var[:, self._nb_latents * self._latent_dim:])



class Decoder(nn.Module):
    def __init__(self, nb_attributes, latent_dim, hidden_dim, output_dim, cntx_latent_dim=0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim * nb_attributes + cntx_latent_dim, hidden_dim), nn.ReLU(),
                                 #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))
    def forward(self, z):
        #z = torch.view(z.shape[0], -1)
        return self.net(z)


class AttrDecoder(nn.Module):
    def __init__(self, nb_attributes, latent_dim, hidden_dim, output_dim=1, attr_in_01=False):
        super().__init__()
        self.latent_to_hidden = BlockLinear(latent_dim, hidden_dim, nb_attributes)
        self.hidden_to_out = BlockLinear(hidden_dim, output_dim, nb_attributes)
        if attr_in_01:
            self.hidden_to_out = nn.Sequential(self.hidden_to_out, nn.Sigmoid())

    def forward(self, x):
       x = torch.relu(self.latent_to_hidden(x))
       return self.hidden_to_out(x)

# for i in range(self.latent_to_hidden.weight.shape[0]):
#     self.latent_to_hidden.weight.data[i] = torch.ones_like(self.latent_to_hidden.weight[i])*(i+1)
#
#
#
class ADisVAE(Model):
    def __init__(self, input_dim, hidden_dim, nb_attributes, latent_dim, cntx_latent_dim=0, attr_in_01=False):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._cntx_latent_dim = cntx_latent_dim
        self._nb_attributes = nb_attributes
        self.encoder = Encoder(nb_attributes, input_dim, hidden_dim, latent_dim, cntx_latent_dim)
        self.decoder = Decoder(nb_attributes, latent_dim, hidden_dim, input_dim, cntx_latent_dim)
        self.attr_decoder = AttrDecoder(nb_attributes, latent_dim, int(latent_dim / 2), 1, attr_in_01=attr_in_01)
        self.cntx_attr_decoder = nn.Sequential(GradientReversal(1),
                                               nn.Linear(cntx_latent_dim, cntx_latent_dim//2), nn.ReLU(),
                                               nn.Linear(cntx_latent_dim//2, nb_attributes))
        if attr_in_01:
            self.cntx_attr_decoder = nn.Sequential(self.cntx_attr_decoder, nn.Sigmoid())

        self._normal_dist = tdist.Normal(0, 0)

    def forward(self, x, a_mask=None):
        mu, var = self.encode(x)
        # if a_mask is not None:
        #     z_mu[0] = self.mask(z_mu[0], a_mask)
        #     z_var[0] = self.mask(z_var[0], a_mask)
        z_attr, z_cntx = self.sample_z(mu, var, a_mask)
        x1 = self.decode_x(z_attr, z_cntx, a_mask)
        a1 = self.decode_a(z_attr)
        a_cntx = self.cntx_attr_decoder(z_cntx)
        return x1, a1, a_cntx, mu, var

    def encode(self, x):
        (mu_attr, mu_cntx), (var_attr, var_cntx) = self.encoder(x)
        return (mu_attr, mu_cntx), (var_attr, var_cntx)

    def sample_z(self, mu, var, a_mask=None):
        mu = torch.cat(mu, 1)
        var = torch.cat(var, 1)
        std = torch.exp(var / 2)
        eps = torch.randn_like(std)  # eps = self._normal_dist.sample(std.shape).to(self.device)
        z = eps.mul(std).add_(mu)
        return z[:, :self._nb_attributes*self._latent_dim], z[:, self._nb_attributes*self._latent_dim:]

    def decode_x(self, z_attr, z_cntx, a_mask=None):
        if a_mask is not None:
            z_attr = self.mask(z_attr, a_mask)
        z = torch.cat((z_attr, z_cntx), dim=1)
        return self.decoder(z)

    def decode_a(self, z_attr):
        return self.attr_decoder(z_attr)


    def mask(self, z_attr, a_mask):
        a_mask = torch.repeat_interleave(a_mask, self._latent_dim, dim=-1)
        return z_attr * a_mask

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
            x_gen = self.decoder(z)
        return x_gen

    def fc2conv(self, x):
        return x.view(x.shape[0], -1, self._nb_attributes)
    def conv2fc(self, x):
        return x.view(x.shape[0], -1)


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

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(),
                                 nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


#%% TSNE

def tsne_plot(data: Tuple[torch.FloatTensor], markers=['o', 'X', '.'], alphas=[1, 0.5, 0.3],
              nb_pca_components=None, append_title='', legend=False, savepath=None, figsize=(14, 8),  dpi=250,
              **kwargs):
    if isinstance(data[0], torch.Tensor):
        data = [data]
    Xs = [NP(d[0]) for d in data]
    X = np.concatenate(Xs, axis=0)

    Ys = [NP(d[1]) for d in data]
    Y =  np.concatenate(Ys, axis=0)
    ### PCA
    if nb_pca_components is not None:
        pca = PCA(n_components=nb_pca_components)
        print("Fitting PCA and transforming...")
        pca.fit(X)
        X = pca.transform(X)

    ### TSNE FIT
    print("Fitting TSNE and transforming...")
    tsne_data = TSNE(n_components=2, **kwargs).fit_transform(X)

    b = len(Xs[0])
    Ts = [tsne_data[:b]]
    for x in (Xs[1:]):
        new_b = b+len(x)
        Ts.append(tsne_data[b:new_b])
        b = new_b

    # PLOT TSNE
    from matplotlib import pyplot as plt
    import seaborn as sns
    fig = plt.figure(figsize=figsize, dpi=dpi)
    title = f'tsne {append_title}'
    fig.suptitle(title, fontsize=16)
    for T, Y, m, a in zip(Ts, Ys, markers[:len(Ts)], alphas[:len(Ts)]):
        sns.scatterplot(x=T[:, 0], y=T[:, 1], hue=Y[:, 0], marker=m, alpha=a, palette=sns.color_palette("hls", len(set(Y[:,0]))),
                        legend=legend)

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, title + '.png'))
    else:
        plt.show()

#%% TEST
def zs_test(vae: ADisVAE,
            test_classifier: nn.Module,
            train_dict,
            test_dict,
            nb_gen_class_samples=100,
            adapt_epochs: int = 5,
            adapt_lr: float = .0001,
            adapt_wd: float=0,
            adapt_bs: int = 128,
            mask_attr_key='class_attr_bin',
            plot_tsne=False,
            plot_tsne_l2_norm=True,
            seed=None,
            device=None):
    test_A_mask = torch.tensor(test_dict[mask_attr_key]).float().to(device)
    test_A = torch.tensor(test_dict['class_attr']).float().to(device)
    test_feats = torch.tensor(test_dict['feats']).float()
    test_labels = torch.tensor(test_dict['labels']).long()
    nb_test_classes = test_dict['class_attr_bin'].shape[0]
    test_loader =  DataLoader(TensorDataset(test_feats, test_labels), batch_size=adapt_bs, num_workers=2, shuffle=True)

    def test_on_test():
        from sklearn.neighbors import KNeighborsClassifier

        nn_cls = KNeighborsClassifier(n_neighbors=1)
        nn_cls.fit(NP(test_A), np.arange(test_A.shape[0]))

        losses, preds, y_trues = [], [], []
        nn_preds = []
        test_classifier.eval()
        for x, y in test_loader:
            x = x.to(device); y = y.to(device)
            logit = test_classifier(x)
            # TODO: reconstruct attribute a' and use 1-NN for classification (DAP baseline)
            mu, var = vae.encode(x)
            z_attr, z_cntx = vae.sample_z(mu, var, None)
            a1 = vae.decode_a(z_attr)

            #l1 = F.l1_loss(test_A[y], a1)
            #logit = test_classifier(x1)
            loss = NN.cross_entropy(logit, y)
            losses.append(loss.detach().cpu());
            preds.append(logit.argmax(dim=1));
            nn_preds.append(nn_cls.predict(NP(a1)))
            y_trues.append(y)
        preds = torch.cat(preds); y_trues = torch.cat(y_trues);
        nn_preds = np.concatenate(nn_preds, axis=0)
        unseen_loss = torch.stack(losses).mean()
        unseen_acc = np.mean([metrics.recall_score(NP(y_trues), NP(preds), labels=[k], average=None) for k in sorted(set(NP(y_trues)))])
        unseen_acc_nn = np.mean([metrics.recall_score(NP(y_trues), nn_preds, labels=[k], average=None) for k in sorted(set(NP(y_trues)))])
        return unseen_loss, unseen_acc, unseen_acc_nn


    ######## OPTIMIZER PREPARATION ###########
    test_classifier.to(device)
    optim = torch.optim.Adam(test_classifier.parameters(), lr=adapt_lr, weight_decay=adapt_wd)


    ######## DATA GENERATION/PREPARATION ###########
    if seed is not None:
        set_seed(seed)
    # adapt_ds = ProbabilisticInfiniteDataset(nb_gen_class_samples * nb_test_classes, lambda x: vae.encoder(x),
    #                                         train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
    #                                         test_dict['class_attr_bin'],
    #                                         device=device, use_context=False)

    adapt_ds = InfiniteDataset(nb_gen_class_samples * nb_test_classes, vae.encode,
                               train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
                               test_dict['class_attr_bin'], use_context=True, device=device)
    adapt_loader = DataLoader(adapt_ds, batch_size=adapt_bs, num_workers=0, shuffle=True, )


    ######## TRAINING NEW CLASSIFIER ###########
    best_unseen_acc = 0
    best_classifier_state = None
    unseen_accs = []
    unseen_dap_accs = []
    for ep in range(adapt_epochs):
        rec_x = []
        rec_y = []
        preds = []
        y_trues = []
        losses = []
        frnk_feats = []
        test_classifier.train()
        for attr_emb, cntx_emb, y in adapt_loader:
            mu_attr, var_attr = attr_emb[0].to(device), attr_emb[1].to(device)
            mu_cntx, var_cntx = cntx_emb[0].to(device), cntx_emb[1].to(device)
            y = y.to(device)
            z_attr, z_cntx = vae.sample_z((mu_attr, mu_cntx), (var_attr, var_cntx), test_A_mask[y])
            x = vae.decode_x(z_attr, z_cntx, test_A_mask[y])
            frnk_feats.append(NP(x))
            optim.zero_grad()
            rec_x.append(x.detach().cpu())
            rec_y.append(y.detach().cpu())
            logit = test_classifier(x)
            loss = NN.cross_entropy(logit, y)
            loss.backward()
            optim.step()
            losses.append(loss.detach().cpu()); preds.append(logit.argmax(dim=1)); y_trues.append(y)
        preds = torch.cat(preds); y_trues = torch.cat(y_trues); acc = (y_trues == preds).float().mean()
        classifier_losses = torch.stack(losses).mean()
        unseen_loss, unseen_acc, unseen_acc_nn = test_on_test()
        unseen_accs.append(unseen_acc)
        unseen_dap_accs.append(unseen_acc_nn)
        print(f"Classifier adaptation - Epoch {ep + 1}/{adapt_epochs}:   Loss={classifier_losses:1.5f}    Acc={acc:1.4f}  -  uLoss="
              f"{unseen_loss:1.5f}   uAcc={unseen_acc:1.5f}   1-NN-uACC={unseen_acc_nn}")
        if unseen_acc > best_unseen_acc:
            best_unseen_acc = unseen_acc
            best_classifier_state = test_classifier.state_dict()
        test_classifier.load_state_dict(best_classifier_state)

    if plot_tsne:
        rec_x = torch.cat(rec_x)
        rec_y = torch.cat(rec_y)[:, None]
        if plot_tsne_l2_norm:
            rec_x /= (torch.norm(rec_x, p=2, dim=1)[:, None]+np.finfo(float).eps)
            test_feats /= (torch.norm(test_feats, p=2, dim=1)[:, None]+np.finfo(float).eps)
        tsne_plot([(test_feats, test_labels[:, None]), (rec_x, rec_y)], append_title=f"", legend='brief')

    return unseen_accs, np.mean(unseen_dap_accs)


def zs_test_exemplarNN(vae: ADisVAE,
                       train_dict,
                       test_dict,
                       nb_gen_class_samples=100,
                       adapt_bs: int = 128,
                       mask_attr_key='class_attr_bin',
                       plot_tsne=False,
                       plot_tsne_l2_norm=True,
                       seed=None,
                       device=None):
    test_A_mask = torch.tensor(test_dict[mask_attr_key]).float().to(device)
    test_A = torch.tensor(test_dict['class_attr']).float().to(device)
    test_feats = torch.tensor(test_dict['feats']).float()
    test_labels = torch.tensor(test_dict['labels']).long()
    nb_test_classes = test_dict['class_attr_bin'].shape[0]
    test_loader =  DataLoader(TensorDataset(test_feats, test_labels), batch_size=adapt_bs, num_workers=2, shuffle=True)

    ######## DATA GENERATION/PREPARATION ###########
    if seed is not None:
        set_seed(seed)

    adapt_ds = InfiniteDataset(nb_gen_class_samples * nb_test_classes, vae.encode,
                               train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
                               test_dict['class_attr_bin'], use_context=True, device=device)
    adapt_loader = DataLoader(adapt_ds, batch_size=adapt_bs, num_workers=0, shuffle=True, )

    print("Predict class exemplars")
    rec_x = []
    rec_y = []
    for attr_emb, cntx_emb, y in adapt_loader:
        mu_attr, var_attr = attr_emb[0].to(device), attr_emb[1].to(device)
        mu_cntx, var_cntx = cntx_emb[0].to(device), cntx_emb[1].to(device)
        y = y.to(device)
        z_attr, z_cntx = vae.sample_z((mu_attr, mu_cntx), (var_attr, var_cntx))
        x = vae.decode_x(z_attr, z_cntx, test_A_mask[y])
        rec_x.append(x.detach().cpu().numpy())
        rec_y.append(y.detach().cpu().numpy())
    rec_x = np.vstack(rec_x)
    rec_y = np.concatenate(rec_y)

    exemplars = np.zeros([nb_test_classes, rec_x.shape[1]])
    for y in range(nb_test_classes):
        idx = np.argwhere(rec_y==y)[:, 0]
        exemplars[y] = np.mean(rec_x[idx], axis=0)

    exemplars_labels = np.arange(nb_test_classes)

    print("Fit Exemplar-NN")
    from sklearn.neighbors import KNeighborsClassifier
    #nn_cls = KNeighborsClassifier(n_neighbors=1)
    #nn_cls.fit(exemplars, exemplars_labels)


    print("Predict test labels")
    nn_preds = []
    y_trues = []

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        bs = len(x)
        nb_masks = test_A_mask.shape[0]
        # TODO: encode->decode using all the possible attributes and get the most probable using some criterion (the nearest at one of the exemplar?)
        mu, var = vae.encode(x)

        z_attr, z_cntx = vae.sample_z(mu, var)
        per_mask_z_attr = torch.repeat_interleave(z_attr, nb_masks, dim=0)
        per_mask_z_cntx = torch.repeat_interleave(z_cntx, nb_masks, dim=0)
        a_mask = test_A_mask.repeat([bs, 1])
        a_mask = a_mask.repeat_interleave(vae._latent_dim, dim=-1)
        per_mask_z_attr = per_mask_z_attr * a_mask
        x1 = vae.decode_x(per_mask_z_attr, per_mask_z_cntx)

        # The first nb_masks elements is x[0] masked by the nb_masks different masks,
        # The element from nb_masks to nb_masks*2-1 is x[1] masked by the nb_masks different masks, etc...
        x1.view(bs, nb_masks, 2048)

        from scipy.spatial.distance import cdist
        dists = cdist(NP(x1), exemplars, metric='euclidean')
        dists = np.reshape(dists, [bs, nb_masks, nb_masks])

        # Per each element in batch, get the "best" masking (i.e. get the mask resulting in the shortest distance to one of the exemplars)
        argmin = np.argmin(dists, axis=-1)

        # Now we create the [bs x nb_exemplar] matrix
        M = np.array([[dists[i, j, argmin[i, j]] for j in range(nb_masks)] for i in range(bs)])
        preds = np.argmin(M, axis=-1)
        #preds = nn_cls.predict(NP(x1))
        nn_preds.append(preds)
        y_trues.append(y)
    y_trues = torch.cat(y_trues)
    nn_preds = np.concatenate(nn_preds, axis=0)
    unseen_acc_enn = np.mean([metrics.recall_score(NP(y_trues), nn_preds, labels=[k], average=None) for k in sorted(set(NP(y_trues)))])

    if plot_tsne:
        rec_x = torch.cat(rec_x)
        rec_y = torch.cat(rec_y)[:, None]
        if plot_tsne_l2_norm:
            rec_x /= (torch.norm(rec_x, p=2, dim=1)[:, None]+np.finfo(float).eps)
            test_feats /= (torch.norm(test_feats, p=2, dim=1)[:, None]+np.finfo(float).eps)
        tsne_plot([(test_feats, test_labels[:, None]), (rec_x, rec_y)], append_title=f"", legend='brief')

    print(f"Exemplar-NN acc: {unseen_acc_enn:2.5f}")
    return unseen_acc_enn

#%% MAIN
from torch import optim

def frankenstain_batch(z_mu, z_var, A, y):
    z_mu_attr, z_mu_cntx = (t.detach() for t in z_mu)
    z_var_attr, z_var_cntx = (t.detach() for t in z_var)
    bs = z_mu_attr.shape[0]
    nb_attributes = A.shape[1]

    attr_encodings = (z_mu_attr.view(bs, nb_attributes, -1), z_var_attr.view(bs, nb_attributes, -1))
    cntx_encodings = (z_mu_cntx, z_var_cntx)

    K = A[y].long()#.cpu()

    idx = torch.stack([(K * torch.stack([torch.randperm(bs) for _ in range(nb_attributes)]).t()).argmax(0).int() for _ in range(bs)]).long()
    frnk_attr_encs = []
    frnk_cntx_encs = []
    for i in range(len(attr_encodings)):
        a_enc = attr_encodings[i][idx, torch.arange(nb_attributes)] * K[:,:,None].repeat(1, 1, attr_encodings[i].shape[-1]).float().to(z_mu_attr.device)
        frnk_attr_encs.append(a_enc)
        frnk_cntx_encs.append(cntx_encodings[i])
    return frnk_attr_encs, frnk_cntx_encs


import csv
from datetime import datetime


def run_exp(attr_latent_dim=16,  # 8
            cntx_latent_dim=512,
            hidden_dim=1536,
            nb_epochs=200,

            g_lr=.0001,
            d_lr=.00001,
            g_wd=.001,
            d_wd=.001,
            l2_norm_alpha=10,

            adapt_epochs=7,
            adapt_lr=.003,
            adapt_wd=.001,
            adapt_gen_samples=500,
            test_period=3,

            adversarial=False,
            frnk_cycle=False,
            frnk_reconstruct=False,
            attr_in_01=False,
            cce=False,

            gpu=0,
            wseed=42,
            twseed=42,
            tfseed=42,
            write_csv=True):
    D_params = dict(locals())

    if write_csv:
        timestamp = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime('%Y-%m-%d %H:%M:%S')
        csv_fname = f'csv/7_3 - {timestamp}.csv'
        with open(csv_fname, 'w') as csv_f:
            csv_writer = csv.DictWriter(csv_f, D_params.keys())
            csv_writer.writeheader()
            csv_writer.writerow(D_params)
            csv_writer = None

    #device, train, test_unseen, test_seen, val = init_exp(gpu=0,seed=None, dataset='AWA2', attr_in_01=ATTR_IN_01)
    device, train, test_unseen, test_seen, val = init_exp(gpu=gpu, seed=wseed, dataset='AWA2', attr_in_01=attr_in_01)

    # device, train, test_unseen, test_seen, val = init_exp(gpu=0, seed=42, dataset='CUB', attr_in_01=ATTR_IN_01)
    # nb_gen_class_samples_train=1400
    # nb_gen_class_samples_test=400
    # adapt_epochs = 15
    # adapt_lr = .003
    # adapt_wd = .0001
    # test_period = 10

    feats_dim = len(train['feats'][0])
    nb_train_classes, nb_attributes = train['class_attr'].shape
    nb_test_classes, _ = test_unseen['class_attr'].shape

    A_mask = torch.tensor(train['class_attr_bin']).float()
    A = torch.tensor(train['class_attr']).float()

    attribute_masks = torch.tensor(train['attr_bin']).float()
    attributes = torch.tensor(train['attr']).float()
    trainset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long(), attribute_masks, attributes)


    vae = ADisVAE(feats_dim, hidden_dim, nb_attributes=nb_attributes,
                  latent_dim=attr_latent_dim, cntx_latent_dim=cntx_latent_dim,
                  attr_in_01=attr_in_01).to(device)
    discr = Discriminator(feats_dim).to(device)
    classifier = nn.Sequential(L2Norm(l2_norm_alpha, norm_while_test=True),
                               nn.Linear(feats_dim, nb_train_classes)).to(device)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)

    opt_g = optim.Adam(list(vae.parameters()) + list(classifier.parameters()), lr=g_lr, weight_decay=g_wd)
    opt_d = optim.Adam(discr.parameters(), lr=d_lr, weight_decay=d_wd)

    # opt_g = optim.SGD(list(vae.parameters()) + list(classifier.parameters()), lr=g_lr,
    #                   weight_decay=g_wd, momentum=.9, nesterov=False)
    # opt_d = optim.SGD(discr.parameters(), lr=d_lr, weight_decay=d_wd, momentum=.8)


    def get_gen_loss(mu, var, x1, x, a1=None, ac=None, a=None, lossbox=None):
        mu = torch.cat(mu, 1)
        var = torch.cat(var, 1)
        L = lossbox if lossbox is not None else LossBox()
        KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())  # kl divergence loss
        L.append('KLD', KLD / len(x));
        RCLX = F.l1_loss(x1, x, reduction='sum')  # reconstruction loss
        L.append('RCL-X', RCLX / len(x));
        loss = 1 * RCLX + 1 * KLD
        if a1 is not None:
            RCLA = F.l1_loss(a1, a, reduction='sum')  # reconstruction loss
            L.append('RCL-A', RCLA / len(x))
            loss+= 50*RCLA
        if ac is not None:
            RCLA_cntx = F.l1_loss(ac, a, reduction='sum')
            L.append('RCL-A-CNTX', RCLA_cntx / len(x))
            loss += 10*RCLA_cntx

        L.append('total', loss / x.shape[0])
        L.append('total', loss / x.shape[0])
        return loss

    adversarial_loss = torch.nn.BCELoss()

    D_results = []
    for ep in range(nb_epochs):
        # if TRAIN_W_FRNK:
        #     frnk_ds = InfiniteDataset(nb_gen_class_samples_train * nb_test_classes, lambda x: vae.encoder(x),
        #                               train['feats'], train['labels'], train['class_attr_bin'], train['class_attr_bin'],
        #                               device=device, use_context=False)
        #lossbox = vae.fit_loader(train_loader, optimizer, alpha=(2000, 0, 1))
        vae.train()
        L = LossBox()
        frnkL = LossBox()
        for i, (x, y, a_mask, a) in enumerate(train_loader):
            x = x.to(vae.device); a = a.to(vae.device); a_mask = a_mask.to(vae.device)

            real = torch.ones(x.shape[0], 1).requires_grad_(False).to(device)
            fake = torch.zeros(x.shape[0], 1).requires_grad_(False).to(device)

            real_mean = x.mean()
            real_std = x.std(0).mean()

            opt_g.zero_grad()
            x1, a1, ac, z_mu, z_var = vae.forward(x, a_mask)
            reconstr_mean = x1.mean()
            reconstr_std = x1.std(0).mean()
            gen_loss = get_gen_loss(z_mu, z_var, x1, x, a1, ac, a, L)

            if adversarial:
                fooling_loss = adversarial_loss(discr(x1), real)
                L.append('G-fooling', fooling_loss)
                gen_loss = gen_loss + fooling_loss

            if cce:
                logits1 = classifier(x1)
                cce1 = NN.cross_entropy(logits1, y.to(device))
                gen_loss += cce1
                L.append('CCE1', cce1)

            gen_loss.backward()
            opt_g.step()

            if frnk_cycle or frnk_reconstruct:
                (frnk_mu_attr, frnk_var_attr), (frnk_mu_cntx, frnk_var_cntx) = frankenstain_batch(z_mu, z_var, A_mask, y)
                frnk_mu_attr = frnk_mu_attr.view(frnk_mu_attr.shape[0], -1).to(device)
                frnk_var_attr = frnk_var_attr.view(frnk_var_attr.shape[0], -1).to(device)
                frnk_mu_cntx, frnk_var_cntx = frnk_mu_cntx.to(device), frnk_var_cntx.to(device)

                opt_g.zero_grad()
                frnk_mu = (frnk_mu_attr, frnk_mu_cntx) # torch.cat((z_mu_attr, z_mu_cntx), dim=1)
                frnk_var = (frnk_var_attr, frnk_var_cntx) #torch.cat((z_var_attr, z_var_cntx), dim=1)
                frnk_attr, frnk_cntx = vae.sample_z(frnk_mu, frnk_var, a_mask)
                frnk_attr = frnk_attr.to(device)
                frnk_cntx = frnk_cntx.to(device)
                x1_frnk = vae.decode_x(frnk_attr, frnk_cntx, a_mask)
                a1_frnk = vae.decode_a(frnk_attr)
                ac_frnk = vae.cntx_attr_decoder(frnk_cntx)

                (enc_frnk_mu_attr, enc_frnk_mu_cntx), (enc_frnk_var_attr, enc_frnk_var_cntx) = vae.encode(x1_frnk)

                frnk_mean = x1_frnk.mean()
                frnk_std = x1_frnk.var(0).mean()


                loss = 0
                if frnk_reconstruct:
                    RCL_FRNK = F.l1_loss(x1_frnk, x, reduction='sum')  # reconstruction loss
                    frnkL.append("RCL-frnk", RCL_FRNK / len(x1_frnk))
                    loss += 10*RCL_FRNK

                if frnk_cycle:
                    CYCLE_mu = NN.l1_loss(enc_frnk_mu_attr, frnk_mu_attr.to(device), reduction='sum')
                    CYCLE_var = NN.l1_loss(enc_frnk_var_attr, frnk_var_attr.to(device), reduction='sum')
                    CYCLE = (CYCLE_mu + CYCLE_var) / 2
                    loss += 5*CYCLE
                    frnkL.append("CYCLE-frnk", CYCLE / len(x1_frnk))

                loss.backward()


                # frnk_loss = get_gen_loss(z_mu, z_var, x1_frnk, x, a1_frnk, ac_frnk, a, frnkL)
                # if ADVERSARIAL:
                #     fooling_loss = adversarial_loss(discr(x1_frnk), real)
                #     L.append('frankG-fooling', fooling_loss)
                #     frnk_loss = (frnk_loss + fooling_loss)
                # frnk_loss.backward()
                opt_g.step()

            if adversarial:
                opt_d.zero_grad()
                real_loss = adversarial_loss(discr(x), real)
                fake_loss = adversarial_loss(discr(x1.detach()), fake)
                L.append('D-real', real_loss)
                L.append('D-fake', fake_loss)
                if frnk_reconstruct or frnk_cycle:
                    fake_loss_frnk = adversarial_loss(discr(x1_frnk.detach()), fake)
                    L.append('frankG-fooling', fake_loss_frnk)
                    d_loss = (real_loss + (fake_loss + fake_loss_frnk)/2) / 2
                else:
                    d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                opt_d.step()

        print(f"real:         {real_mean:2.4f}  +/- {real_std:2.4f}")
        print(f"reconstr:     {reconstr_mean:2.4f}  +/- {reconstr_std:2.4f}")
        if frnk_reconstruct or frnk_cycle:
            print(f"frankenstain: {frnk_mean:2.4f}  +/- {frnk_std:2.4f}")
        print("\n")

        print(f'====> Epoch: {ep+1}')
        print(L)
        if frnk_reconstruct or frnk_cycle:
            print(frnkL)

        if (ep+1)%test_period == 0:
            # Test on unseen-test-set:
            set_seed(twseed)
            adapt_classifier = nn.Sequential(#nn.Linear(feats_dim, feats_dim), nn.ReLU(),
                                             L2Norm(l2_norm_alpha, norm_while_test=True),
                                             nn.Linear(feats_dim, nb_test_classes))
            # accs, dap_acc = zs_test(vae, adapt_classifier, train, test_unseen,
            #                         nb_gen_class_samples=adapt_gen_samples, adapt_epochs=adapt_epochs, adapt_lr=adapt_lr, adapt_wd=adapt_wd,
            #                         adapt_bs=128, mask_attr_key='class_attr_bin', device=device,
            #                         plot_tsne=False, plot_tsne_l2_norm=False, seed=tfseed)

            enn_acc = zs_test_exemplarNN(vae, train, test_unseen, adapt_bs=128, mask_attr_key='class_attr_bin', device=device,
                                               plot_tsne=False, plot_tsne_l2_norm=False, seed=tfseed)


            #D_results.append({'epoch': ep, 'best-acc': np.max(accs), 'zs-accs': accs, 'mean-acc': np.mean(accs), 'DAP-1NN-acc': dap_acc})
            D_results.append({'epoch': ep, 'enn-acc': enn_acc})

            if write_csv:
                with open(csv_fname, 'a') as csv_f:
                    if csv_writer is None:
                        csv_f.write("\n\n")
                        csv_writer = csv.DictWriter(csv_f, D_results[-1].keys())
                        csv_writer.writeheader()
                    else:
                        csv_writer = csv.DictWriter(csv_f, D_results[-1].keys())
                    csv_writer.writerow(D_results[-1])

    return D_results, D_params



import os
if __name__ == '__main__':
    g_wd = 0 # 1e-5
    d_wd = 0 # 1e-4
    adapt_wd = 0 # 1e-4

    def csv_fname_fn():
        return f'csv/7_3 - {datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%Y-%m-%d %H:%M:%S")} - SUMMARY.csv'
    csv_fname = None

    PARAMS = []
    BEST_RESULTS = []
    for adversarial in (False, ):
        for adversarial in (False, ):
            for frnk_cycle in (False, ):
                for frnk_reconstruct in (False, ):
                #for wd in (0, .00001, .001):
                    for seed, tfseed in (#[40, 40], [40, 41], [40, 42],
                                         [41, 40], [41, 41], [41, 42],
                                         [42, 40], [42, 41], [42, 42], ):

                        D_results, D_Params = run_exp(attr_latent_dim=32,  # 8
                                                      cntx_latent_dim=512,
                                                      hidden_dim=2048,
                                                      nb_epochs=100,

                                                      #g_lr=.000001,
                                                      g_lr=.0001,
                                                      d_lr=.00001,
                                                      g_wd=g_wd,
                                                      d_wd=d_wd,
                                                      l2_norm_alpha=10,

                                                      adapt_epochs=10,
                                                      adapt_lr=.001,
                                                      adapt_wd=adapt_wd,
                                                      adapt_gen_samples=500,
                                                      test_period=1,
                                                      adversarial=adversarial,
                                                      frnk_cycle=frnk_cycle,
                                                      frnk_reconstruct=frnk_reconstruct,

                                                      attr_in_01=False,
                                                      wseed=seed,
                                                      twseed=seed,
                                                      tfseed=tfseed,)
                        best_test_accs = [d_result['best-acc'] for d_result in D_results]
                        best = np.argmax(best_test_accs)
                        BEST_RESULTS.append(D_results[best])
                        PARAMS.append(D_Params)

                        D = {**D_results[best], **D_Params}

                        if csv_fname is None:
                            csv_fname = csv_fname_fn()
                            with open(csv_fname, 'w') as csv_f:
                                csv_writer = csv.DictWriter(csv_f, D.keys())
                                csv_writer.writeheader()
                                csv_writer.writerow(D)
                        else:
                            old_name = csv_fname
                            csv_fname = csv_fname_fn()
                            os.rename(old_name, csv_fname)
                            with open(csv_fname, 'a') as csv_f:
                                csv_writer = csv.DictWriter(csv_f, D.keys())
                                csv_writer.writerow(D)








#%% TRASHCAN



