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
        self.var = nn.Linear(hidden_dim, latent_dim * nb_attributes)
        self._latent_dim = latent_dim
        self._nb_latents = nb_attributes

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        return self.mu(hidden), self.var(hidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim, nb_attributes, hidden_dim, output_dim):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim * nb_attributes, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        #z = torch.view(z.shape[0], -1)
        z1 = F.relu(self.latent_to_hidden(z))
        return self.hidden_to_out(z1)

class AttrDecoder(nn.Module):
    def __init__(self, latent_dim, nb_attributes, hidden_dim, output_dim=1):
        super().__init__()
        self.latent_to_hidden = BlockLinear(latent_dim, hidden_dim, nb_attributes)
        self.hidden_to_out = BlockLinear(hidden_dim, output_dim, nb_attributes)
        # self.latent_to_hidden = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        # self.hidden_to_out = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
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
        self.decoder = Decoder(latent_dim, nb_attributes, hidden_dim, input_dim)
        self.attr_decoder = AttrDecoder(latent_dim, nb_attributes, int(latent_dim/2),  1)
        self._normal_dist = tdist.Normal(0, 0)

    def forward(self, x, a_mask=None):
        z_mu, z_var = self.encoder(x)
        z = self.sample_z(z_mu, z_var)
        # decode
        if a_mask is not None:
            x1 = self.decoder(self.mask(z, a_mask))
        else:
            x1 = self.decoder(z)
        # attr decode
        #z_conv = self.fc2conv(z)
        a1 = self.attr_decoder(z_mu)
        #a1 = self.conv2fc(a1)
        return x1, a1, z_mu, z_var


    def compute_loss(self, x, reconstructed_x, a, reconstructed_a, mean, log_var):
        RCLX = F.mse_loss(reconstructed_x, x, reduction='none').sum(dim=1).mean(dim=0)   # reconstruction loss
        #RCLX = 1-F.cosine_similarity(reconstructed_x, x).mean()   # reconstruction loss
        RCLA = F.mse_loss(reconstructed_a, a, reduction='none').sum(dim=1).mean(dim=0)    # reconstruction loss
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  # kl divergence loss
        return RCLX, RCLA, KLD

    def fit_loader(self, train_loader, optimizer, alpha=(1, 1, 4)):
        self.train()
        L = LossBox()
        for i, (x, a_mask, a) in enumerate(train_loader):
            x = x.to(self.device)
            a = a.to(self.device)
            a_mask = a_mask.to(self.device)
            optimizer.zero_grad()
            x1, a1, z_mu, z_var = self.forward(x, a_mask)
            RCLx, RCLa, KLD = self.compute_loss(x, x1, a, a1, z_mu, z_var,)
            loss = alpha[0]*RCLx + alpha[1]*RCLa + alpha[2]*KLD
            loss.backward()
            optimizer.step()
            L.append('RCL-X', RCLx); L.append('RCL-A', RCLa); L.append('KLD', KLD); L.append('loss', loss/x.shape[0])
        return L

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

    def mask(self, z, a):
        a_mask = torch.repeat_interleave(a, self._latent_dim, dim=-1)
        return z*a_mask

    def sample_z(self, z_mu, z_var):
        # sample from the distribution having latent parameters z_mu, z_var reparameterize
        std = torch.exp(z_var/2)#*0.1
        eps = torch.randn_like(std)#*0.1
        # std = torch.exp(z_var/2)
        # eps = self._normal_dist.sample(std.shape).to(self.device)
        z = eps.mul(std).add_(z_mu)
        return z

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

    def tsne_plot(self, data: Tuple[torch.FloatTensor], markers=['o', 'X', '.'], alphas=[1, 0.5, 0.3],
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


#%% TEST

def zs_test(vae: ADisVAE,
            test_classifier: nn.Module,
            train_dict,
            test_dict,
            nb_gen_class_samples=100,
            adapt_epochs: int = 5,
            adapt_lr: float = .0001,
            adapt_bs: int = 128,
            attrs_key='class_attr_bin',
            plot_tsne=False,
            plot_tsne_l2_norm=True,
            device=None):
    test_A = torch.tensor(test_dict[attrs_key]).float().to(device)
    test_feats = torch.tensor(test_dict['feats']).float()
    test_labels = torch.tensor(test_dict['labels']).long()
    nb_test_classes = test_dict['class_attr_bin'].shape[0]
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
    adapt_ds = ProbabilisticInfiniteDataset(nb_gen_class_samples * nb_test_classes, lambda x: vae.encoder(x),
                                            train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
                                            test_dict['class_attr_bin'],
                                            device=device, use_context=False)

    adapt_loader = DataLoader(adapt_ds, batch_size=adapt_bs, num_workers=0, shuffle=True, )



    ######## TRAINING NEW CLASSIFIER ###########
    test_classifier.to(device)
    optim = torch.optim.Adam(test_classifier.parameters(), lr=adapt_lr)
    best_unseen_acc = 0
    best_classifier_state = None


    for ep in range(adapt_epochs):
        rec_x = []
        rec_y = []
        preds = []
        y_trues = []
        losses = []
        test_classifier.train()
        for emb, y in adapt_loader:
            mu, var = emb
            mu = mu.to(device); var = var.to(device); y = y.to(device)
            z = vae.sample_z(mu, var)
            x = vae.decoder(vae.mask(z, test_A[y]))
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
        unseen_loss, unseen_acc = test_on_test()

        print(f"Classifier adaptation - Epoch {ep + 1}/{adapt_epochs}:   Loss={classifier_losses:1.5f}    Acc={acc:1.4f}  -  uLoss="
              f"{unseen_loss:1.5f}   uAcc={unseen_acc:1.5f}")
        if unseen_acc > best_unseen_acc:
            best_unseen_acc = unseen_acc
            best_classifier_state = test_classifier.state_dict()
        test_classifier.load_state_dict(best_classifier_state)
    if plot_tsne:
        # vae.tsne_plot([(test_feats, test_labels[:,None]), (torch.cat(rec_x), torch.cat(rec_y)[:,None])], append_title=f"",
        #             perplexity=50, num_neighbors=6, early_exaggeration=12)
        rec_x = torch.cat(rec_x)
        rec_y = torch.cat(rec_y)[:, None]
        if plot_tsne_l2_norm:
            rec_x /= (torch.norm(rec_x, p=2, dim=1)[:, None]+np.finfo(float).eps)
            test_feats /= (torch.norm(test_feats, p=2, dim=1)[:, None]+np.finfo(float).eps)

        vae.tsne_plot([(test_feats, test_labels[:, None]), (rec_x, rec_y)], append_title=f"")

    return best_unseen_acc  # , best_classifier_state
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

        lossbox = vae.fit_loader(train_loader, optimizer, alpha=(0.01, 10, .001)) # RCLx + RCLa + KLD
        #lossbox = vae.fit_loader(train_loader, optimizer, alpha=(.001,50,2))
        print(f'====> Epoch: {ep+1}')
        print(lossbox)

        if (ep+1)%10 == 0:
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

