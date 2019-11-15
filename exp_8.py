import math
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
from torch.nn import functional as F, Parameter

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
class LocallyConnected(nn.Module):
    __constants__ = ['bias']

    def __init__(self, in_features, nb_outs, bias=True):
        super(LocallyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = (nb_outs, in_features)
        self.weight = Parameter(torch.Tensor(nb_outs, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        out = self.weight * input + self.bias
        return out

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class MyModel(Model):
    def __init__(self, input_dim, nb_attributes):
        super().__init__()
        self._input_dim = input_dim
        self._nb_attributes = nb_attributes
        self.W11 = Parameter(torch.Tensor(nb_attributes, input_dim))
        self.W12 = Parameter(torch.Tensor(nb_attributes, input_dim))
        self.b1 = None # Parameter(torch.Tensor(input_dim))
        self.W2 = Parameter(torch.Tensor(nb_attributes, input_dim))
        self.b2 = None #Parameter(torch.Tensor(nb_attributes))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W11, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W12, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        if self.b1 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W11)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b1, -bound, bound)
        if self.b2 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W2)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b2, -bound, bound)

    def encode_x(self, x):
        bs = x.shape[0]
        x = x[:, None, :].repeat(1, self._nb_attributes, 1)
        # M1 = torch.sigmoid(self.M1[None, :, :]).repeat(bs, 1, 1)
        # M1 = torch.relu(torch.tanh(self.M1[None, :, :])).repeat(bs, 1, 1)
        # M1 = torch.relu(self.M1[None, :, :]).repeat(bs, 1, 1)
        W11 = self.W11[None, :, :].repeat(bs, 1, 1)
        W12 = self.W12[None, :, :].repeat(bs, 1, 1)
        EX1 = torch.relu(W11 * x)
        EX = torch.relu(W12 * EX1)
        return EX

    def encode_a(self, EX):
        bs = EX.shape[0]
        # M2 = torch.sigmoid(self.M2[None, :, :]).repeat(bs, 1, 1)
        # M2 = torch.relu(torch.tanh(self.M2[None, :, :])).repeat(bs, 1, 1)
        # M2 = torch.relu(self.M2[None, :, :]).repeat(bs, 1, 1)
        W2 = self.W2[None, :, :].repeat(bs, 1, 1)
        EA = W2 * EX
        return EA

    def mask(self, EX, a_mask):
        return EX * a_mask[:, :, None].repeat(1, 1, EX.shape[-1])

    def decode_x(self, EX, a_mask=None):
        if a_mask is not None:
            EX = self.mask(EX, a_mask)
        return EX.sum(dim=1)

    def decode_a(self, EA):
        return EA.sum(dim=2)

    def forward(self, x, a_mask=None):
        EX = self.encode_x(x)
        EA = self.encode_a(EX)
        return self.decode_x(EX, a_mask), self.decode_a(EA)




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
def zs_test(model: MyModel,
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
    test_loader =  DataLoader(TensorDataset(test_feats, test_labels), batch_size=adapt_bs,
                              num_workers=0, pin_memory=True,shuffle=True)

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
            EX = model.encode_x(x)
            EA = model.encode_a(EX)
            a1 = model.decode_a(EA)

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

    adapt_ds = InfiniteDataset(nb_gen_class_samples * nb_test_classes, lambda x: model.encode_x(x).cpu(),
                               train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
                               test_dict['class_attr_bin'], use_context=False, reshape=False, device=device)
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
        for attr_emb, y in adapt_loader:
            EX = attr_emb[0].to(device)
            x = model.decode_x(EX)
            y = y.to(device)
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
              f"{unseen_loss:1.4f}  uAcc={unseen_acc:1.4f}  1-NN-uACC={unseen_acc_nn:1.4f}")
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

def frankenstain_batch(EX, A):
    EX = EX.detach()
    bs = EX.shape[0]
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



if __name__ == '__main__':
    nb_epochs = 60
    ATTR_IN_01=False
    ADVERSARIAL=False
    TRAIN_W_FRNK=False
    #device, train, test_unseen, test_seen, val = init_exp(gpu=0,seed=None, dataset='AWA2', attr_in_01=ATTR_IN_01)
    device, train, test_unseen, test_seen, val = init_exp(gpu=1, seed=42, dataset='AWA2', attr_in_01=ATTR_IN_01)
    nb_gen_class_samples_train = 1400
    nb_gen_class_samples_test=500
    adapt_epochs = 5
    adapt_lr = .001
    adapt_wd = 0 #.00001
    test_period = 2

    g_lr = .001
    d_lr = .001
    g_wd = 0 #.00001
    d_wd = 0 #.00001


    feats_dim = len(train['feats'][0])
    nb_train_classes, nb_attributes = train['class_attr'].shape
    nb_test_classes, _ = test_unseen['class_attr'].shape

    A_mask = torch.tensor(train['class_attr_bin']).float()
    A = torch.tensor(train['class_attr']).float()

    attribute_masks = torch.tensor(train['attr_bin']).float()
    attributes = torch.tensor(train['attr']).float()
    trainset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long(), attribute_masks, attributes)


    net = MyModel(feats_dim, nb_attributes).to(device)
    classifier = nn.Sequential(L2Norm(10, norm_while_test=True),
                               nn.Linear(feats_dim, nb_train_classes)).to(device)
    discr = Discriminator(feats_dim)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)

    # opt_g = optim.Adam(list(net.parameters()) + list(classifier.parameters()), lr=g_lr, weight_decay=g_wd)
    # opt_d = optim.Adam(discr.parameters(), lr=d_lr, weight_decay=d_wd)

    opt_g = optim.SGD(list(net.parameters()) + list(classifier.parameters()), lr=g_lr, weight_decay=g_wd, momentum=.7)
    opt_d = optim.SGD(discr.parameters(), lr=d_lr, weight_decay=d_wd, momentum=.7)

    def get_gen_loss(x, a, x1=None, a1=None, ac=None, lossbox=None):
        L = lossbox if lossbox is not None else LossBox()
        loss = 0
        if x1 is not None:
            RCLX = F.mse_loss(x1, x)  # reconstruction loss
            # L.append('RCL-X', RCLX / len(x))
            L.append('RCL-X', RCLX)
            loss += 1*RCLX

        if a1 is not None:
            RCLA = F.mse_loss(a1, a)  # reconstruction loss
            # L.append('RCL-A', RCLA / len(x))
            L.append('RCL-A', RCLA)
            loss += 1*RCLA

        if ac is not None:
            RCLA_cntx = F.l1_loss(ac, a, reduction='sum')
            L.append('RCL-A-CNTX', RCLA_cntx / len(x))
            loss += 10*RCLA_cntx

        #L.append('total', loss / x.shape[0])
        L.append('total', loss)
        return loss


    adversarial_loss = torch.nn.BCELoss()

    for ep in range(nb_epochs):

        net.train()
        L = LossBox()
        frnkL = LossBox()
        for i, (x, y, a_mask, a) in enumerate(train_loader):
            x = x.to(net.device); a = a.to(net.device); a_mask = a_mask.to(net.device)

            real = torch.ones(x.shape[0], 1).requires_grad_(False).to(device)
            fake = torch.zeros(x.shape[0], 1).requires_grad_(False).to(device)

            real_mean = x.mean()
            real_std = x.std(0).mean()

            opt_g.zero_grad()
            x1, a1 = net.forward(x, a_mask)
            reconstr_mean = x1.mean()
            reconstr_std = x1.std(0).mean()
            gen_loss = get_gen_loss(x, a, x1, a1, lossbox=L)
            if ADVERSARIAL:
                fooling_loss = adversarial_loss(discr(x1), real)
                L.append('G-fooling', fooling_loss)
                gen_loss = gen_loss + fooling_loss

            logits1 = classifier(x1)
            cce1 = NN.cross_entropy(logits1, y.to(device))
            gen_loss += .001 * cce1
            L.append('CCE1', cce1)

            gen_loss.backward()
            opt_g.step()

            if TRAIN_W_FRNK:
                (frnk_mu_attr, frnk_var_attr), (frnk_mu_cntx, frnk_var_cntx) = frankenstain_batch(z_mu, z_var, A_mask)
                frnk_mu_attr = frnk_mu_attr.view(frnk_mu_attr.shape[0], -1).to(device)
                frnk_var_attr = frnk_var_attr.view(frnk_var_attr.shape[0], -1).to(device)
                frnk_mu_cntx, frnk_var_cntx = frnk_mu_cntx.to(device), frnk_var_cntx.to(device)

                opt_g.zero_grad()
                frnk_mu = (frnk_mu_attr, frnk_mu_cntx) # torch.cat((z_mu_attr, z_mu_cntx), dim=1)
                frnk_var = (frnk_var_attr, frnk_var_cntx) #torch.cat((z_var_attr, z_var_cntx), dim=1)
                frnk_attr, frnk_cntx = net.sample_z(frnk_mu, frnk_var, a_mask)
                frnk_attr = frnk_attr.to(device)
                frnk_cntx = frnk_cntx.to(device)
                x1_frnk = net.decode_x(frnk_attr, frnk_cntx, a_mask)
                a1_frnk = net.decode_a(frnk_attr)
                ac_frnk = net.cntx_attr_decoder(frnk_cntx)

                (enc_frnk_mu_attr, enc_frnk_mu_cntx), (enc_frnk_var_attr, enc_frnk_var_cntx) = net.encode(x1_frnk)

                frnk_mean = x1_frnk.mean()
                frnk_std = x1_frnk.var(0).mean()

                CYCLE_mu = NN.l1_loss(enc_frnk_mu_attr, frnk_mu_attr.to(device), reduction='sum')
                CYCLE_var = NN.l1_loss(enc_frnk_var_attr, frnk_var_attr.to(device), reduction='sum')
                CYCLE = (CYCLE_mu + CYCLE_var) / 2
                RCL_FRNK = F.l1_loss(x1_frnk, x, reduction='sum')  # reconstruction loss
                loss = 0 * RCL_FRNK + 1 * CYCLE
                loss.backward()

                frnkL.append("RCL-frnk", RCL_FRNK / len(x1_frnk))
                frnkL.append("CYCLE-frnk", CYCLE/len(x1_frnk))

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

        if (ep+1)%test_period == 0:
            # Test on unseen-test-set:
            adapt_classifier = nn.Sequential(#nn.Linear(feats_dim, feats_dim), nn.ReLU(),
                                       L2Norm(10, norm_while_test=True),
                                       nn.Linear(feats_dim, nb_test_classes))
            zs_test(net, adapt_classifier, train, test_unseen,
                    nb_gen_class_samples=nb_gen_class_samples_test, adapt_epochs=adapt_epochs, adapt_lr=adapt_lr, adapt_wd=adapt_wd,
                    adapt_bs=128, mask_attr_key='class_attr_bin', device=device,
                    plot_tsne=True, plot_tsne_l2_norm=False)



    # TODO:

#%% TRASHCAN



