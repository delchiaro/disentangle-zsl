import os
from dataclasses import dataclass

from sortedcontainers import SortedSet
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


@dataclass
class LossMult:
    rec_x:float = 1.
    rec_a:float = 1.
    rec_ac:float = 1.
    rec_ad:float = 1.
    cls_x:float = .01
    cls_x1:float = .01
    cls_cntx:float = .001

@dataclass
class AutoencoderHiddens:
    encX_Z: List = None
    encZ_X: List = None
    encZ_KA: List = None
    encZ_KC: List = None
    decK_Z: List = None
    decKA_A: List = None
    decKA_disA: List = None
    decKC_disC: List = None


GRL = GradientReversal

class NoneLayer(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device=device

    def to(self, *args, **kwargs):
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device
        return self

    def forward(self, x):
        return torch.tensor([]).float().to(self.device)


class Autoencoder(nn.Module):
    def __init__(self, feats_dim, z_dim, ka_dim, kc_dim, nb_attributes, hiddens=AutoencoderHiddens(),
                 conv1by1=True, use_context=True):
        super().__init__()
        self.feats_dim = feats_dim
        self.z_dim = z_dim
        self.ka_dim = ka_dim
        self.kc_dim = kc_dim = kc_dim if use_context else 0
        self.nb_attributes = nb_attributes
        self._use_1by1_conv = conv1by1
        self.use_context = use_context

        hid_act = nn.ReLU
        self.encX_Z = get_fc_net(feats_dim, hiddens.encX_Z, z_dim, hid_act(), hid_act())
        self.decZ_X = get_fc_net(z_dim, hiddens.encZ_X, feats_dim, hid_act(), hid_act())

        self.encZ_KA = get_fc_net(z_dim, hiddens.encZ_KA, ka_dim * nb_attributes, hid_act(), hid_act())
        self.encZ_KC = get_fc_net(z_dim, hiddens.encZ_KC, kc_dim, hid_act(), hid_act()) if use_context else NoneLayer()
        self.decK_Z = get_fc_net(ka_dim * nb_attributes + kc_dim, hiddens.decK_Z, z_dim, hid_act(), hid_act())

        if conv1by1:
            self.decKA_A = get_1by1_conv1d_net(ka_dim, hiddens.decKA_A, 1, hid_act(), nn.Sigmoid())
            self.decKA_disA = nn.Sequential(GRL(1), get_1by1_conv1d_net(ka_dim, hiddens.decKA_disA, nb_attributes - 1, hid_act(), nn.Sigmoid()))
        else:
            self.decKA_A = get_block_fc_net(nb_attributes, ka_dim, hiddens.decKA_A, 1, hid_act(), nn.Sigmoid())
            self.decKA_disA = nn.Sequential(GRL(1), get_block_fc_net(nb_attributes, ka_dim, hiddens.decKA_disA, nb_attributes - 1, hid_act(), nn.Sigmoid()))
        self.decKC_disC = nn.Sequential(GRL(1), get_fc_net(kc_dim, hiddens.decKC_disC, nb_attributes, hid_act(), nn.Sigmoid()))  if use_context else NoneLayer()

    def to(self, *args, **kwargs):
        self.decKC_disC.to(*args, **kwargs)
        self.encZ_KC.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def mask_ka(self, ka, a_mask):
        return ka * a_mask.repeat_interleave(self.ka_dim, dim=-1)

    def encode(self, x, a_mask=None):
        z = self.encX_Z(x)
        ka = self.encZ_KA(z)
        if a_mask is not None:
            ka = self.mask_ka(ka, a_mask)
        kc = self.encZ_KC(z)
        return ka, kc

    def decode(self, ka, kc, a_mask=None):
        ka = self.mask_ka(ka, a_mask) if a_mask is not None else ka
        z1 = self.decK_Z(torch.cat([ka, kc], dim=1))
        return self.decZ_X(z1)

    def attr_decode(self, ka, a_mask=None):
        ka = self.mask_ka(ka, a_mask) if a_mask is not None else ka
        if self._use_1by1_conv:
            ka = ka.view(ka.shape[0], self.ka_dim, self.nb_attributes)  # to be used with 1b1 conv nets
        a1 = self.decKA_A(ka).squeeze(dim=1)
        return a1

    def dis_attr_decode(self, ka, a_mask=None):
        ka = self.mask_ka(ka, a_mask) if a_mask is not None else ka
        if self._use_1by1_conv:
            ka = ka.view(ka.shape[0], self.ka_dim, self.nb_attributes)  # to be used with 1b1 conv nets
        ad = self.decKA_disA(ka)
        if self._use_1by1_conv:
            ad = ad.transpose(-1, -2)  # to be used with 1b1 conv nets
        else:
            ad = ad.view(ad.shape[0], self.nb_attributes, -1) # to be used with block-linear net
        return ad

    def cntx_attr_decode(self, kc):
        return self.decKC_disC(kc)

    def forward(self, x, a_mask=None, attr_dis=False):
        ka, kc = self.encode(x)
        x1 = self.decode(ka, kc, a_mask)
        a1 = self.attr_decode(ka)
        ac = self.cntx_attr_decode(kc)
        ad = self.dis_attr_decode(ka) if attr_dis else None
        return ka, kc, x1, a1, ac, ad


class Classifier(nn.Module):
    def __init__(self, input_dim, nb_classes, hidden_units=None, hidden_activations=nn.ReLU()):
        super().__init__()
        self.input_dim = input_dim
        self.nb_classes = nb_classes
        self.hidden_units = hidden_units
        self.net = get_fc_net(input_dim, hidden_units, nb_classes, hidden_activations, out_activation=None)

    @property
    def linear_classifier_input_dim(self):
        return self.input_dim if self.hidden_units is None else self.hidden_units[-1]

    def _get_linear_classifier(self, new_nb_classes=None):
        if new_nb_classes is not None:
            self.nb_classes = new_nb_classes
        return nn.Linear(self.linear_classifier_input_dim, out_features=self.nb_classes, bias=True)

    def reset_linear_classifier(self, new_nb_classes=None):
        device = self.net[-1].weight.device
        self.net[-1] = self._get_linear_classifier(new_nb_classes).to(device)
        return self

    def forward(self, x):
        return self.net(x)


def disentangle_loss(autoencoder: Autoencoder, classifier: Classifier, attr_enc, cntx_enc, label, all_mask_attrs):
    nb_classes = all_mask_attrs.shape[0]
    bs = len(attr_enc)
    attr_enc_exp = torch.repeat_interleave(attr_enc, repeats=nb_classes, dim=0)
    cntx_enc_exp = torch.repeat_interleave(cntx_enc, repeats=nb_classes, dim=0)
    # label_exp = interlaced_repeat(label, nb_classes, dim=0)
    all_attrs_exp = all_mask_attrs.repeat([bs, 1])
    #
    decoded = autoencoder.decode(attr_enc_exp, cntx_enc_exp, all_attrs_exp)
    logits = classifier(decoded)
    t = torch.tensor([[t] for t in list(range(nb_classes)) * bs]).to(attr_enc.device)
    logits_diags = torch.gather(logits, 1, t).view(bs, nb_classes)
    return NN.cross_entropy(logits_diags, label)


class Model:
    def __init__(self,
                 feats_dim, nb_attributes, nb_train_classes, nb_test_classes,
                 z_dim, ka_dim, kc_dim,
                 autoencoder_hiddens=AutoencoderHiddens(), cls_hiddens=(2048,), cntx_cls_hiddens=None,
                 conv1by1=True, use_context=True,
                 device=None):
        self.autoencoder = Autoencoder(feats_dim, z_dim, ka_dim, kc_dim, nb_attributes, autoencoder_hiddens, conv1by1, use_context).to(device)
        self.cntx_classifier = nn.Sequential(GradientReversal(1), Classifier(kc_dim, nb_train_classes, cntx_cls_hiddens)).to(device)
        self.classifier = Classifier(feats_dim, nb_train_classes, cls_hiddens).to(device)
        self.test_classifier = Classifier(feats_dim, nb_train_classes, (2048,)).to(device)
        if nb_test_classes is not None:
            self.test_classifier.reset_linear_classifier(nb_test_classes)
        self.acc = -1
        self.device = device

    def load(self, state_dir='states', epoch=None):
        from os.path import join
        f = join(state_dir, 'model_best.pt') if epoch is None else join(state_dir, f'model_epoch_{epoch:03d}.pt')
        print(f"Loading model {f}")
        state = torch.load(f)
        self.autoencoder.load_state_dict(state['autoencoder'])
        self.classifier.load_state_dict(state['classifier'])
        self.cntx_classifier.load_state_dict(state['cntx_classifier'])
        self.test_classifier.load_state_dict(state['test_classifier'])
        self.acc = state['adapt_acc']
        return self

    def save(self, state_dir='states', epoch=None, acc=None):
        from os.path import join
        if not os.path.isdir(state_dir):
            os.mkdir(state_dir)
        fpath = join(state_dir, 'model_best.pt') if epoch is None else join(state_dir, f'model_epoch_{epoch:03d}.pt')
        self.acc = self.acc if acc is None else acc
        torch.save({'autoencoder': self.autoencoder.state_dict(),
                    'classifier': self.classifier.state_dict(),
                    'cntx_classifier': self.cntx_classifier.state_dict(),
                    'test_classifier': self.test_classifier.state_dict(),
                    'adapt_acc': self.acc,
                    }, fpath)

    @classmethod
    def show(cls, state_dir='states'):
        states_files = sorted(os.listdir(state_dir))
        best_state = torch.load(os.path.join(state_dir, 'best_models.pt'))
        best_acc = best_state['adapt_acc']
        for state_f in states_files:
            state = torch.load(f'states/{state_f}')
            acc = state['adapt_acc']
            try:
                ep = int(state_f.split('_')[-1].split('.')[0])
                print(f"epoch {ep:03d}  -   acc={acc:2.4f}" + (' --- BEST ' if acc == best_acc else ''))
            except:
                continue



def diag_nondiag_idx(rows_cols):
    diag_idx = SortedSet([(i, i) for i in range(rows_cols)])
    all_idx = SortedSet([tuple(l) for l in torch.triu_indices(rows_cols, rows_cols, -rows_cols).transpose(0, 1).numpy().tolist()])
    non_diag_idx = all_idx.difference(diag_idx)
    non_diag_idx = torch.Tensor(non_diag_idx).transpose(0, 1).long()
    return diag_idx, non_diag_idx

#%% EXP
def exp(model: Model, train, test_unseen, bs=128, nb_epochs=100, a_lr=.00001, loss_mult=LossMult(),
        pretrain_cls_epochs=5, c_lr=.0001,
        bin_attrs=False, bin_masks=False,
        masking=True, attr_disentangle=False, train_frankenstain=False,
        early_stop_patience=None,

        test_period=1, test_epochs=10, test_lr=.0001, test_gen_samples=300,
        infinite_dataset=True,
        test_tsne=False,
        verbose=2,
        state_dir='states/exp_5',
        save_states=False):
    attrs_key = 'class_attr_bin' if bin_attrs else 'class_attr'
    mask_attrs_key = 'class_attr_bin' if bin_masks else 'class_attr'

    train_dataset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

    nb_train_classes, nb_attributes = train[attrs_key].shape
    nb_test_classes, _ = test_unseen[attrs_key].shape
    A = torch.tensor(train[attrs_key]).float().to(model.device)
    A_mask = torch.tensor(train[mask_attrs_key]).float().to(model.device)

    diag_idx, non_diag_idx = diag_nondiag_idx(nb_attributes)

    if save_states:
        acc = gen_zsl_test(model.autoencoder, model.test_classifier, train, test_unseen, nb_gen_class_samples=test_gen_samples,
                           infinite_dataset=infinite_dataset,
                           attrs_key=attrs_key, attrs_mask_key=mask_attrs_key,
                           use_masking=masking,
                           adapt_epochs=test_epochs, adapt_lr=test_lr, device=model.device)
        model.save(state_dir, epoch=0, acc=acc)

    # *********** CLASSIFIER PRE-TRAINING
    c_opt = torch.optim.Adam(model.classifier.parameters(), lr=c_lr)
    for i in range(pretrain_cls_epochs):
        L = LossBox('4.5f')
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(model.device), y.to(model.device)

            c_opt.zero_grad()
            logits = model.classifier(x)
            l_cls = NN.cross_entropy(logits, y)
            l_cls.backward()
            c_opt.step()
            L.append('cls_x', l_cls);
        if verbose >= 3:
            print(L)


    # autoencoder_params = list(model.autoencoder.parameters())
    flatten = lambda l: [item for sublist in l for item in sublist]
    decoder_params = flatten([list(m.parameters()) for m in [model.autoencoder.decK_Z, model.autoencoder.decZ_X,
                                                             model.autoencoder.decKA_A, model.autoencoder.decKC_disC,
                                                             model.autoencoder.decKA_disA]])
    encoder_params = flatten([list(m.parameters()) for m in
                              [model.autoencoder.encX_Z, model.autoencoder.encZ_KA, model.autoencoder.encZ_KC]])

    # *********** AUTOENCODER TRAINING
    opt = torch.optim.Adam(list(model.autoencoder.parameters()) +
                           list(model.classifier.parameters()) +
                           list(model.cntx_classifier.parameters()), lr=a_lr)

    a_opt = torch.optim.Adam(encoder_params# + decoder_params
                             + list(model.classifier.parameters())
                             + list(model.cntx_classifier.parameters()), lr=a_lr)
    d_opt = torch.optim.Adam(decoder_params + list(model.classifier.parameters()) + list(model.cntx_classifier.parameters()),
                             lr=a_lr*10)
    zero_loss = torch.tensor([0.]).mean().to(model.device)
    def compute_loss(lossBox:LossBox, x, a, y, x1, a1, ac=None, ad=None, logits=None, logits1=None, logits_cntx=None):
        l_rec_x = NN.mse_loss(x1, x)
        l_rec_a1 = NN.mse_loss(a1, a)
        l_rec_ac = NN.mse_loss(ac, a) if ac is not None else zero_loss
        l_rec_ad = NN.mse_loss(ad, a_non_diag)  if ad is not None else zero_loss
        l_cls1 = NN.cross_entropy(logits1, y)  if logits1 is not None else zero_loss
        l_cls_cntx = NN.cross_entropy(logits_cntx, y)  if logits_cntx is not None else zero_loss
        l_cls = NN.cross_entropy(logits, y)  if logits is not None else zero_loss
        # l_cls_contrastive = disentangle_loss(autoencoder, classifier, ka, kc, y, A_mask)

        l = loss_mult.rec_x * l_rec_x + \
            loss_mult.rec_a * l_rec_a1 + \
            loss_mult.rec_ac * l_rec_ac + \
            loss_mult.rec_ad * l_rec_ad + \
            loss_mult.cls_x * l_cls + \
            loss_mult.cls_x1 * l_cls1 + \
            loss_mult.cls_cntx * l_cls_cntx
        # l = 5*l_rec_x + 5*l_rec_a1 + 1*l_rec_ad + 1*l_rec_ac + .01*l_cls + .01*l_cls1 + .001*l_cls_cntx
        # l += l_cls_contrastive * .01
        lossBox.append('x', l_rec_x);
        lossBox.append('a1', l_rec_a1);
        lossBox.append('ad', l_rec_ad);
        lossBox.append('ac', l_rec_ac);
        lossBox.append('cls_x', l_cls);
        lossBox.append('cls_x1', l_cls1);
        lossBox.append('l_cls_cntx', l_cls_cntx);
        return l

    train_accs, valid_accs, test_accs = [], [], []
    best_patience_score = 0
    best_acc = 0
    patience = early_stop_patience
    l_rec_ad = torch.tensor([0.])
    fac = ac = flogits_cntx = logits_cntx = None

    for ep in range(nb_epochs):
        if verbose >= 2:
            print("\n")
            print(f"Running Epoch {ep + 1}/{nb_epochs}")

        if train_frankenstain:
            frnk_dataset = InfiniteDataset(len(train_dataset), model.autoencoder.encode, train['feats'], train['labels'], train['class_attr_bin'],
                                           train['class_attr_bin'], use_context=model.autoencoder.use_context, device=model.device)

        L = LossBox('4.5f')
        FL = LossBox('4.5f')
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(model.device), y.to(model.device)
            a = A[y]
            a_mask = A_mask[y]  if masking else None # Masking with continuous attributes seems to work better!

            a_dis = a[:, None, :].repeat(1, nb_attributes, 1)
            a_non_diag = a_dis[:, non_diag_idx[0], non_diag_idx[1]].reshape(a.shape[0], nb_attributes, nb_attributes - 1)
            # a_diag = a_dis[:, diag_idx[0], diag_idx[1]].reshape(a.shape[0], nb_attributes)

            opt.zero_grad()
            #a_opt.zero_grad()
            #ka, kc, x1, a1, ac, ad = model.autoencoder.forward(x, a_mask if masking else None, attr_disentangle)
            ka, kc = model.autoencoder.encode(x)
            #x1 = model.autoencoder.decode(ka, kc, a_mask)
            x1 = model.autoencoder.decode(ka, kc, a) # using a instead of a_mask!
            a1 = model.autoencoder.attr_decode(ka, a_mask)
            ad = model.autoencoder.dis_attr_decode(ka, a_mask) if attr_disentangle else None
            logits = model.classifier(x)
            logits1 = model.classifier(x1)
            if len(kc.shape) > 1:
                logits_cntx = model.cntx_classifier(kc)
                ac = model.autoencoder.cntx_attr_decode(kc)
            l = compute_loss(L, x, a, y, x1, a1, ac, ad, logits, logits1, logits_cntx)
            l.backward()
            opt.step()
            #a_opt.step()

            if train_frankenstain:
                opt.zero_grad()
                #a_opt.zero_grad()
                #d_opt.zero_grad()
                fka, fkc, fy = [t.to(model.device).detach() for t in frnk_dataset.get_items_with_class(y)]
                #frnk_a_mask = A_mask[fy]  if masking else None
                #fx1 = model.autoencoder.decode(fka, fkc, a_mask)
                fx1 = model.autoencoder.decode(fka, fkc, a) # using a instead of a_mask!
                fa1 = model.autoencoder.attr_decode(fka, a_mask)
                fad = model.autoencoder.dis_attr_decode(fka, a_mask) if attr_disentangle else None
                flogits1 = model.classifier(fx1)
                if len(kc.shape) > 1:
                    fac = model.autoencoder.cntx_attr_decode(fkc)
                    flogits_cntx = model.cntx_classifier(fkc)
                fl = compute_loss(FL, x, a, y, fx1, fa1, fac, fad, None, flogits1, flogits_cntx)
                fl.backward()
                opt.step()
                #d_opt.step()
            # L.append('l_cls_contrastive', l_cls_contrastive);

        if verbose >= 3:
            print(L)
            if train_frankenstain:
                print(FL)


        if test_period is not None and (ep + 1) % test_period == 0:
            if test_tsne:
                tsne(model, [test_unseen, train], train, infinite_dataset=True, target='feats')
            acc = gen_zsl_test(model.autoencoder, model.test_classifier, train, test_unseen, nb_gen_class_samples=test_gen_samples,
                               infinite_dataset=infinite_dataset,
                               attrs_key=attrs_key, attrs_mask_key=mask_attrs_key,
                               use_masking=masking,
                               adapt_epochs=test_epochs, adapt_lr=test_lr, device=model.device)
            test_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                print(f'New best accuracy: {best_acc:2.4f}')
                if save_states:
                    model.save(state_dir, epoch=None, acc=acc)

            else:
                print(f'Current accuracy: {acc:2.4f}')
                print(f'Old best accuracy: {best_acc:2.4f}')
            if save_states:
                model.save(state_dir, ep, acc)

        if patience is not None:
            #pscore = valid_accs[-1] if val is not None else test_accs[-1]
            pscore = test_accs[-1]
            if best_patience_score < pscore:
                best_patience_score = pscore
                patience = early_stop_patience
            else:
                patience -= 1
            if patience <= 0:
                if verbose >= 1:
                    print(f'--- Early Stopping ---     Best Score: {best_patience_score},  Best Test: {np.max(test_accs)}')
                break
    return train_accs, valid_accs, test_accs

#%% TEST
def gen_zsl_test(autoencoder: Autoencoder,
                 test_classifier: Classifier,
                 train_dict,
                 zsl_unseen_test_dict,
                 nb_gen_class_samples=100,
                 adapt_epochs: int = 5,
                 adapt_lr: float = .0001,
                 adapt_bs: int = 128,
                 attrs_key='class_attr',
                 attrs_mask_key='class_attr',  # 'class_attr_bin'
                 use_masking=True,
                 infinite_dataset=True,
                 device=None):
    nb_new_classes = len(zsl_unseen_test_dict['class_attr_bin'])
    A = torch.tensor(zsl_unseen_test_dict[attrs_key]).float().to(device)
    A_mask = torch.tensor(zsl_unseen_test_dict[attrs_mask_key]).float().to(device)
    ######## PREPARE NEW CLASSIFIER ###########
    test_classifier.reset_linear_classifier(nb_new_classes)

    def test_on_test():
        unseen_test_feats = torch.tensor(zsl_unseen_test_dict['feats']).float()
        unseen_test_labels = torch.tensor(zsl_unseen_test_dict['labels']).long()
        dloader = DataLoader(TensorDataset(unseen_test_feats, unseen_test_labels), batch_size=adapt_bs, num_workers=2)
        losses, preds, y_trues = [], [], []
        for X, Y in dloader:
            X = X.to(device);
            Y = Y.to(device)
            # z, ka, kc, z1, x1, a1, ad, ac = net(X)
            logit = test_classifier(X)
            loss = NN.cross_entropy(logit, Y)
            losses.append(loss.detach().cpu());
            preds.append(logit.argmax(dim=1));
            y_trues.append(Y)

        preds = torch.cat(preds);
        y_trues = torch.cat(y_trues);
        unseen_loss = torch.stack(losses).mean()
        unseen_acc = np.mean([metrics.recall_score(NP(y_trues), NP(preds), labels=[k], average=None) for k in sorted(set(NP(y_trues)))])
        return unseen_loss, unseen_acc

    ######## DATA GENERATION/PREPARATION ###########
    def enc_fn(X):
        Z = autoencoder.encX_Z(X)
        return autoencoder.encZ_KA(Z), autoencoder.encZ_KC(Z)

    if infinite_dataset:
        dataset = InfiniteDataset(nb_gen_class_samples * nb_new_classes, enc_fn,
                                  train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
                                  zsl_unseen_test_dict['class_attr_bin'],
                                  use_context=autoencoder.use_context,
                                  device=device)

    else:
        dataset = FrankensteinDataset(nb_gen_class_samples, enc_fn,
                                      train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
                                      zsl_unseen_test_dict['class_attr_bin'], device=device)

    data_loader = DataLoader(dataset, batch_size=adapt_bs, num_workers=0, shuffle=True)

    ######## TRAINING NEW CLASSIFIER ###########
    optim = torch.optim.Adam(test_classifier.net[-1].parameters(), lr=adapt_lr)
    best_unseen_acc = 0
    best_classifier_state = None
    for ep in range(adapt_epochs):
        preds = []
        y_trues = []
        losses = []
        for data in data_loader:
            ka, kc, Y = data[0].to(device), data[1].to(device), data[2].to(device)
            optim.zero_grad()
            X = autoencoder.decode(ka, kc,  A[Y] if use_masking else None) # Using A instead of A_mask
            #X = autoencoder.decode(ka, kc,  A_mask[Y] if use_masking else None)
            logit = test_classifier(X)
            loss = NN.cross_entropy(logit, Y)
            loss.backward()
            optim.step()
            losses.append(loss.detach().cpu())
            preds.append(logit.argmax(dim=1))
            y_trues.append(Y)
        preds = torch.cat(preds)
        y_trues = torch.cat(y_trues)
        acc = (y_trues == preds).float().mean()
        classifier_losses = torch.stack(losses).mean()
        unseen_loss, unseen_acc = test_on_test()
        print(
            f"Classifier adaptation - Epoch {ep + 1}/{adapt_epochs}:   Loss={classifier_losses:1.5f}    Acc={acc:1.4f}  -  uLoss="
            f"{unseen_loss:1.5f}   uAcc={unseen_acc:1.5f}")
        if unseen_acc > best_unseen_acc:
            best_unseen_acc = unseen_acc
            best_classifier_state = test_classifier.state_dict()
        test_classifier.load_state_dict(best_classifier_state)
    return best_unseen_acc  # , best_classifier_state



def tsne(model: Model, test_dicts, train_dict, nb_pca=None, nb_gen_class_samples=200, infinite_dataset=False,
         target='feats', # 'attr_emb', 'cntx_emb'
         append_title='',
         attrs_key='class_attr', mask_attrs_key='class_attr',
         legend=False,
         savepath=None, dpi=250):
    test_dicts = [test_dicts] if isinstance(test_dicts, dict) else test_dicts

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(nrows=len(test_dicts), figsize=(20, 8 * len(test_dicts)), dpi=dpi)
    title = f'tsne {target}{append_title}'
    fig.suptitle(title, fontsize=16)

    for i, data_dict in enumerate(test_dicts):
        nb_classes = len(set(data_dict['labels']))
        test_A = torch.tensor(data_dict[attrs_key]).float().to(model.device)
        test_A_mask = torch.tensor(data_dict[mask_attrs_key]).float().to(model.device)
        X = torch.tensor(data_dict['feats']).float()
        Y = torch.tensor(data_dict['labels']).long()
        dataset = TensorDataset(X, Y)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

        if infinite_dataset:
            frankenstain_dataset = InfiniteDataset(nb_gen_class_samples * nb_classes, model.autoencoder.encode,
                                                   train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
                                                   data_dict['class_attr_bin'],
                                                   use_context=model.autoencoder.use_context,
                                                   device=model.device)
        else:
            frankenstain_dataset = FrankensteinDataset(nb_gen_class_samples, model.autoencoder.encode,
                                                       train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
                                                       data_dict['class_attr_bin'], device=model.device)
        frankenstain_loader = DataLoader(frankenstain_dataset, batch_size=128, num_workers=0, shuffle=True)


        if target in ['cntx_emb', 'attr_emb']:
            KA1 = []; KC1 = []; Y1 = []
            for ka, kc, y in frankenstain_loader:
                #ka = model.autoencoder.apply_mask(ka.to(model.device), test_A_mask[y.to(model.device)])
                KA1.append(ka);  KC1.append(kc); Y1.append(y)

            KA = []; KC = []; Y = []
            for x, y in data_loader:
                ka, kc = model.autoencoder.encode(x.to(model.device))
                #ka = model.autoencoder.apply_mask(ka, test_A_mask[y.to(model.device)])
                KA.append(ka);
                KC.append(kc);
                Y.append(y)
            KA, KC, Y, KA1, KC1, Y1 = (torch.cat(T, dim=0).detach().cpu().numpy() for T in (KA, KC, Y, KA1, KC1, Y1) )
            if target == 'cntx_emb':
                X, X1 = KC, KC1
            elif target is 'attr_emb':
                X, X1 = KA, KA1

        else:
            X1 = []
            Y1 = []
            for data in frankenstain_loader:
                ka, kc, y = data[0].to(model.device), data[1].to(model.device), data[2].to(model.device)
                x1 = model.autoencoder.decode(ka, kc, test_A_mask[y])
                X1.append(x1); Y1.append(y)
            X1 = torch.cat(X1, dim=0); Y1 = torch.cat(Y1, dim=0)
            X, Y, X1, Y1 = (T.detach().cpu().numpy() for T in (X, Y, X1, Y1))


        ### PCA
        if nb_pca is not None:
            pca = PCA(n_components=nb_pca)
            print("Fitting PCA on real images...")
            pca.fit(X)
            print("Transforming real images with PCA...")
            X = pca.transform(X)
            print("Transforming generated images with PCA...")
            X1 = pca.transform(X1)

        ### TSNE FIT
        print("Fitting TSNE and transforming...")
        embeddings = TSNE(n_components=2).fit_transform(np.concatenate([X, X1], axis=0))
        embeddings1 = embeddings[len(X):]
        embeddings = embeddings[:len(X)]

        # PLOT TSNE
        from matplotlib import pyplot as plt
        import seaborn as sns
        sns.scatterplot(x=embeddings[:,0], y=embeddings[:, 1], hue=Y,
                        palette=sns.color_palette("hls", nb_classes), legend=legend, alpha=0.4, ax=axs[i])
        sns.scatterplot(x=embeddings1[:,0], y=embeddings1[:, 1], hue=Y1,
                        palette=sns.color_palette("hls", nb_classes), legend=legend, alpha=1, ax=axs[i])
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, title+'.png'))
    else:
        plt.show()


# %%

def build_model_A(feats_dim: int, nb_attributes: int, nb_train_classes: int, nb_test_classes: int,
                  device: Union[str, None] = None) -> Model:
    return Model(feats_dim, nb_attributes, nb_train_classes, nb_test_classes,
                 z_dim=2048, ka_dim=32, kc_dim=256,  # best kc: 256, 512, 768
                 autoencoder_hiddens=AutoencoderHiddens(), cls_hiddens=(2048,), cntx_cls_hiddens=None,
                 conv1by1=True,
                 use_context=True,
                 device=device)


def build_model_B(feats_dim: int, nb_attributes: int, nb_train_classes: int, nb_test_classes: int,
                  device: Union[str, None] = None) -> Model:
    autoenc_hid = AutoencoderHiddens()
    autoenc_hid.decK_Z = (2048,)
    return Model(feats_dim, nb_attributes, nb_train_classes, nb_test_classes,
                 z_dim=2048, ka_dim=32, kc_dim=256,  # best kc: 256, 512, 768
                 autoencoder_hiddens=autoenc_hid, cls_hiddens=(2048,), cntx_cls_hiddens=None,
                 device=device)


def build_model_C(feats_dim: int, nb_attributes: int, nb_train_classes: int, nb_test_classes: int,
                  device: Union[str, None] = None) -> Model:
    autoenc_hid = AutoencoderHiddens()
    autoenc_hid.encX_Z = (2048,)
    return Model(feats_dim, nb_attributes, nb_train_classes, nb_test_classes,
                 z_dim=2048, ka_dim=32, kc_dim=256,  # best kc: 256, 512, 768
                 autoencoder_hiddens=autoenc_hid, cls_hiddens=(2048,), cntx_cls_hiddens=None,
                 device=device)


build_model_fn = Callable[[int, int, int, int, Union[str, None]], Model]


def init_exp(get_model_fn: build_model_fn, gpu, seed, use_valid=False, download_data=False, preprocess_data=False, dataset='AWA2'):
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
    feats_dim = len(train['feats'][0])

    ### INIT MODEL
    nb_train_classes, nb_attributes = train['class_attr'].shape
    nb_test_classes, _ = test_unseen['class_attr'].shape
    model = get_model_fn(feats_dim, nb_attributes, nb_train_classes, nb_test_classes, device)
    return model, train, test_unseen, test_seen, val

