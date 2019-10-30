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
from utils import init
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

    ######## DATA GENERATION/PREPARATION ###########
    # adapt_ds = ProbabilisticInfiniteDataset(nb_gen_class_samples * nb_test_classes, lambda x: vae.encoder(x),
    #                                         train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
    #                                         test_dict['class_attr_bin'],
    #                                         device=device, use_context=False)
    adapt_ds = InfiniteDataset(nb_gen_class_samples * nb_test_classes, vae.encode,
                               train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
                               test_dict['class_attr_bin'], device=device)
    adapt_loader = DataLoader(adapt_ds, batch_size=adapt_bs, num_workers=0, shuffle=True, )



    ######## TRAINING NEW CLASSIFIER ###########
    test_classifier.to(device)
    optim = torch.optim.Adam(test_classifier.parameters(), lr=adapt_lr, weight_decay=adapt_wd)
    best_unseen_acc = 0
    best_classifier_state = None

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
        # frnk_feats = np.concatenate(frnk_feats, axis=0)
        # frnk_mean = np.mean(frnk_feats, axis=0)
        # real_mean = np.mean(NP(test_feats), axis=0)
        # frnk_std = np.mean(np.std(frnk_feats, axis=0))
        # real_std = np.mean(np.std(NP(test_feats), axis=0))

        print(f"Classifier adaptation - Epoch {ep + 1}/{adapt_epochs}:   Loss={classifier_losses:1.5f}    Acc={acc:1.4f}  -  uLoss="
              f"{unseen_loss:1.5f}   uAcc={unseen_acc:1.5f}   1-NN-uACC={unseen_acc_nn}")
        if unseen_acc > best_unseen_acc:
            best_unseen_acc = unseen_acc
            best_classifier_state = test_classifier.state_dict()
        test_classifier.load_state_dict(best_classifier_state)
    if plot_tsne:
        # tsne_plot([(test_feats, test_labels[:,None]), (torch.cat(rec_x), torch.cat(rec_y)[:,None])], append_title=f"",
        #             perplexity=50, num_neighbors=6, early_exaggeration=12)
        rec_x = torch.cat(rec_x)
        rec_y = torch.cat(rec_y)[:, None]
        if plot_tsne_l2_norm:
            rec_x /= (torch.norm(rec_x, p=2, dim=1)[:, None]+np.finfo(float).eps)
            test_feats /= (torch.norm(test_feats, p=2, dim=1)[:, None]+np.finfo(float).eps)

        tsne_plot([(test_feats, test_labels[:, None]), (rec_x, rec_y)], append_title=f"", legend='brief')

    return best_unseen_acc  # , best_classifier_state
#%% MAIN
from torch import optim


if __name__ == '__main__':
    ATTR_IN_01 = False
    TRAIN_W_FRNK=False
    ADVERSARIAL=True
    MASKING = True

    device, train, test_unseen, test_seen, val = init_exp(gpu=0, seed=42, dataset='AWA2', attr_in_01=ATTR_IN_01)
    #device, train, test_unseen, test_seen, val = init_exp(gpu=0, seed=42, dataset='CUB', attr_in_01=ATTR_IN_01)
    #device, train, test_unseen, test_seen, val = init_exp(gpu=0,seed=None, dataset='AWA2', attr_in_01=ATTR_IN_01)

    feats_dim = len(train['feats'][0])
    nb_train_classes, nb_attributes = train['class_attr'].shape
    nb_test_classes, _ = test_unseen['class_attr'].shape

    A_mask = torch.tensor(train['attr_bin']).float()
    A = torch.tensor(train['attr']).float()
    trainset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long(), A_mask, A)


    vae = ADisVAE(feats_dim, 1536, nb_attributes=nb_attributes,
                  latent_dim=16, cntx_latent_dim=512,
                  attr_in_01=ATTR_IN_01).to(device)
    discr = Discriminator(feats_dim).to(device)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)
    opt_g = optim.Adam(vae.parameters(), lr=.0001, weight_decay=.001)
    opt_d = optim.Adam(discr.parameters(), lr=.00001, weight_decay=.001)


    nb_gen_class_samples=1400


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
            L.append('RCL-A', RCLA / len(x));
            loss+= 50*RCLA
        if ac is not None:
            RCLA_cntx = F.l1_loss(ac, a, reduction='sum')
            L.append('RCL-A-CNTX', RCLA_cntx / len(x))
            loss += 10*RCLA_cntx

        L.append('total', loss / x.shape[0])
        L.append('total', loss / x.shape[0])
        return loss


    adversarial_loss = torch.nn.BCELoss()

    for ep in range(100):
        if TRAIN_W_FRNK:
            frnk_ds = InfiniteDataset(nb_gen_class_samples * nb_test_classes, lambda x: vae.encoder(x),
                                      train['feats'], train['labels'], train['class_attr_bin'], train['class_attr_bin'],
                                      device=device, use_context=False)
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
            if ADVERSARIAL:
                fooling_loss = adversarial_loss(discr(x1), real)
                L.append('G-fooling', fooling_loss)
                gen_loss = gen_loss + fooling_loss
            gen_loss.backward()
            opt_g.step()

            if TRAIN_W_FRNK:
                opt_g.zero_grad()
                (z_mu_attr, z_var_attr), (z_mu_cntx, z_var_cntx), fy = frnk_ds.get_items_with_class(NP(y))
                z_mu = (z_mu_attr, z_mu_cntx) # torch.cat((z_mu_attr, z_mu_cntx), dim=1)
                z_var = (z_var_attr, z_var_cntx) #torch.cat((z_var_attr, z_var_cntx), dim=1)
                z_attr, z_cntx = vae.sample_z(z_mu, z_var, a_mask)
                z_attr = z_attr.to(device)
                z_cntx = z_cntx.to(device)
                x1_frnk = vae.decode_x(z_attr, z_cntx, a_mask)
                a1_frnk = vae.decode_a(z_attr)
                ac_frnk = vae.cntx_attr_decoder(z_cntx)
                frnk_mean = x1_frnk.mean()
                frnk_std = x1_frnk.var(0).mean()

                frnk_loss = get_gen_loss(z_mu, z_var, x1_frnk, x, a1_frnk, ac_frnk, a, frnkL)
                if ADVERSARIAL:
                    fooling_loss = adversarial_loss(discr(x1_frnk), real)
                    L.append('frankG-fooling', fooling_loss)
                    frnk_loss = (frnk_loss + fooling_loss)
                frnk_loss.backward()
                opt_g.step()

            if ADVERSARIAL:
                opt_d.zero_grad()
                real_loss = adversarial_loss(discr(x), real)
                fake_loss = adversarial_loss(discr(x1.detach()), fake)
                L.append('D-real', real_loss)
                L.append('D-fake', fake_loss)
                if TRAIN_W_FRNK:
                    fake_loss_frnk = adversarial_loss(discr(x1_frnk.detach()), fake)
                    L.append('frankG-fooling', fake_loss_frnk)
                    d_loss = (real_loss + (fake_loss + fake_loss_frnk)/2) / 2
                else:
                    d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                opt_d.step()

        print(f"real:         {real_mean:2.4f}  +/- {real_std:2.4f}")
        print(f"reconstr:     {reconstr_mean:2.4f}  +/- {reconstr_std:2.4f}")
        if TRAIN_W_FRNK:
            print(f"frankenstain: {frnk_mean:2.4f}  +/- {frnk_std:2.4f}")
        print("\n")

        print(f'====> Epoch: {ep+1}')
        print(L)
        if TRAIN_W_FRNK:
            print(frnkL)

        if (ep+1)%3 == 0:
            # Test on unseen-test-set:
            classifier = nn.Sequential(#nn.Linear(feats_dim, feats_dim), nn.ReLU(),
                                       L2Norm(10., norm_while_test=True),
                                       nn.Linear(feats_dim, nb_test_classes))
            zs_test(vae, classifier, train, test_unseen,
                    nb_gen_class_samples=500, adapt_epochs=7, adapt_lr=.003, adapt_wd=.001,
                    adapt_bs=128, mask_attr_key='class_attr_bin', device=device,
                    plot_tsne=False, plot_tsne_l2_norm=False)


            # rec_y = trainset_w_cls.tensors[1]
            # mu, var = vae.encoder(trainset_w_cls.tensors[0].to(device))
            # z = vae.sample_z(mu, var)
            # if MASKING:
            #     rec_x = vae.decoder(vae.mask(z.to('cpu'), A_mask).to(device))
            #     a1 = vae.attr_decoder(mu)
            # else:
            #     rec_x = vae.decoder(z)
            # tsne_plot([trainset_w_cls.tensors, (rec_x, rec_y)], append_title=f" - Epoch={ep + 1}",
            #               perplexity = 50, num_neighbors = 6, early_exaggeration = 12)


            # classifier = get_fc_net(feats_dim, hidden_sizes=None, output_size=nb_train_classes)
            # zs_test(vae, classifier, train, train,
            #         nb_gen_class_samples=200, adapt_epochs=1, adapt_lr=.004, adapt_bs=128, mask_attr_key='class_attr_bin', device=device,
            #         plot_tsne=False, plot_tsne_l2_norm=False)


    # TODO:

#%% TRASHCAN



