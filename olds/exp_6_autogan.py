import os
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass

from sortedcontainers import SortedSet
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Callable, Union

import data
from disentangle.dataset_generator import InfiniteDataset, FrankensteinDataset

from disentangle.layers import get_fc_net, GradientReversal, GradientReversalFunction, get_1by1_conv1d_net, IdentityLayer
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
class AutoencoderHiddens:
    encX_Z=None
    encZ_X=None
    encZ_KA=None
    encZ_KC=None
    decK_Z=None
    decKA_A=None
    decKA_disA=None
    decKC_disC=None


class Autoencoder(nn.Module):
    def __init__(self, feats_dim,  nb_attributes, ka_dim, kc_dim, z_dim=None, hiddens: AutoencoderHiddens=AutoencoderHiddens()):
        super().__init__()
        self.feats_dim = feats_dim
        self.nb_attributes = nb_attributes
        self.ka_dim = ka_dim
        self.kc_dim = kc_dim
        self.z_dim = z_dim

        self.encX_Z = get_fc_net(feats_dim, hiddens.encX_Z, z_dim, nn.ReLU(), nn.ReLU()) if z_dim is not None else IdentityLayer()
        self.decZ_X = get_fc_net(z_dim, hiddens.encZ_X, feats_dim, nn.ReLU(), nn.ReLU()) if z_dim is not None else IdentityLayer()

        input_size = z_dim if z_dim is not None else feats_dim
        self.encZ_KA = get_fc_net(input_size, hiddens.encZ_KA, ka_dim * nb_attributes, nn.ReLU(), nn.ReLU())
        self.encZ_KC = get_fc_net(input_size, hiddens.encZ_KC, kc_dim, nn.ReLU(), nn.ReLU())
        self.decK_Z = get_fc_net(ka_dim * nb_attributes + kc_dim, hiddens.decK_Z, input_size, nn.ReLU(), nn.ReLU())

    def encoder_params(self):
        return list(self.encX_Z.parameters()) + list(self.encZ_KA.parameters()) + list(self.encZ_KC.parameters())

    def decoder_params(self):
        return list(self.decK_Z.parameters()) + list(self.decZ_X.parameters())

    def mask_ka(self, ka, a_mask):
        return ka * a_mask.repeat_interleave(self.ka_dim, dim=-1)

    def encode(self, x):
        # Image Encode
        z = self.encX_Z(x)
        ka = self.encZ_KA(z)
        kc = self.encZ_KC(z)
        return ka, kc

    def decode(self, ka, kc, a_mask=None):
        # Image decode
        if a_mask is not None:
            ka_masked = self.mask_ka(ka, a_mask)
            z1 = self.decK_Z(torch.cat([ka_masked, kc], dim=1))
        else:
            z1 = self.decK_Z(torch.cat([ka, kc], dim=1))
        return self.decZ_X(z1)

    def forward(self, x, a_mask=None):
        ka, kc = self.encode(x)  # Image Encode
        x1 = self.decode(ka, kc, a_mask)  # Image decode
        return x1



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

class AttributePredictor(nn.Module):
    def __init__(self, input_dim, nb_attributes, hidden_units=None, hidden_activations=nn.ReLU()):
        super().__init__()
        self.input_dim = input_dim
        self.nb_attributes = nb_attributes
        self.hidden_units = hidden_units
        self.net = get_fc_net(input_dim, hidden_units, nb_attributes, hidden_activations, out_activation=nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, nb_classes, hidden_units=None, hidden_activations=nn.ReLU()):
        super().__init__()
        self.input_dim = input_dim
        self.nb_classes = nb_classes
        self.hidden_units = hidden_units
        self.net = get_fc_net(input_dim, hidden_units, nb_classes, hidden_activations, out_activation=nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class AttrGanModel:
    def __init__(self, feats_dim, nb_attributes, nb_classes,
                 ka_dim, kc_dim, z_dim=None,
                 autoencoder_hiddens=AutoencoderHiddens(),
                 attribute_predictor_hiddens=(512,),
                 classifier_hiddens=None,
                 discriminator_hiddens=(512,),
                 device=None):
        self.autoencoder = Autoencoder(feats_dim, nb_attributes, ka_dim, kc_dim, z_dim, autoencoder_hiddens).to(device)
        self.attr_predictor = AttributePredictor(feats_dim, nb_attributes, attribute_predictor_hiddens, hidden_activations=nn.ReLU()).to(device)
        self.discriminator = get_fc_net(feats_dim, discriminator_hiddens, 1, hidden_activations=nn.ReLU(), out_activation=nn.Sigmoid()).to(device)
        self.classifier = get_fc_net(feats_dim, classifier_hiddens, output_size=nb_classes).to(device)
        self.acc = -1
        self.device = device

    def load(self, state_dir='states', epoch=None):
        from os.path import join
        f = join(state_dir, 'model_best.pt') if epoch is None else join(state_dir, f'model_epoch_{epoch:03d}.pt')
        print(f"Loading model {f}")
        state = torch.load(f)
        self.autoencoder.load_state_dict(state['autoencoder'])
        self.discriminator.load_state_dict(state['discriminator'])
        self.attr_predictor.load_state_dict(state['attr_predictor'])
        if 'adapt_acc' in state.keys():
            self.acc = state['adapt_acc']
        return self

    def save(self, state_dir='states', epoch=None, acc=None):
        from os.path import join
        if not os.path.isdir(state_dir):
            os.mkdir(state_dir)
        fpath = join(state_dir, 'model_best.pt') if epoch is None else join(state_dir, f'model_epoch_{epoch:03d}.pt')
        self.acc = self.acc if acc is None else acc
        torch.save({'autoencoder': self.autoencoder.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'attr_predictor': self.attr_predictor.state_dict(),
                    'adapt_acc': self.acc,
                    }, fpath)



def exp(model: AttrGanModel, train, test_unseen, test_seen=None, val=None,
        bs=128,
        use_masking=False, early_stop_patience=None,
        attrs_key='class_attr', mask_attrs_key='class_attr',
        g_lr=.00001, d_lr=.00001, c_lr=.0001,
        nb_epochs=100, pretrain_cls_epochs=10,
        test_period=1, test_epochs=10, test_lr=.0001, test_gen_samples=300,
        test_use_masking=True, infinite_dataset=True,
        verbose=2, state_dir='states/exp_5'):

    use_valid = True if val is not None else False
    train_dataset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
    frnk_dataset = InfiniteDataset(len(train_dataset), model.autoencoder.encode, train['feats'], train['labels'], train['attr_bin'],
                                   test_unseen['class_attr_bin'], device=model.device)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    frnk_loader =  DataLoader(frnk_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

    nb_train_classes, nb_attributes = train[attrs_key].shape
    feats_dim  = train['feats'].shape[1]

    nb_test_classes, _ = test_unseen[attrs_key].shape
    A = torch.tensor(train[attrs_key]).float().to(model.device)
    A_mask = torch.tensor(train[mask_attrs_key]).float().to(model.device)

    test_classifier = get_fc_net(feats_dim, hidden_sizes=None, output_size=nb_test_classes, hidden_activations=nn.ReLU()).to(model.device)
    acc = gen_zsl_test(model.autoencoder, test_classifier, train, test_unseen, nb_gen_class_samples=test_gen_samples,
                       use_masking=test_use_masking,
                       infinite_dataset=infinite_dataset, adapt_epochs=test_epochs, adapt_lr=test_lr, device=model.device)
    model.save(state_dir, epoch=0, acc=acc)



    # *********** AUTOENCODER TRAINING
    #a_opt = torch.optim.Adam(model.autoencoder.parameters(), lr=a_lr)
    c_opt = torch.optim.Adam(model.classifier.parameters(), lr=c_lr)
    g_opt = torch.optim.Adam(model.autoencoder.decoder_params(), lr=g_lr)
    d_opt = torch.optim.SGD(list(model.autoencoder.parameters()) + list(model.discriminator.parameters()) +
                             list(model.attr_predictor.parameters()) , lr=d_lr)

    adversarial_loss = torch.nn.BCELoss()
    auxiliary_attr_loss = torch.nn.L1Loss()
    auxiliary_cls_loss = torch.nn.CrossEntropyLoss()
    reconstruction_loss = torch.nn.MSELoss()

    train_accs, valid_accs, test_accs = [], [], []
    best_patience_score = 0
    best_acc = 0
    patience = early_stop_patience

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

    for ep in range(nb_epochs):


        if verbose >= 2:
            print("\n")
            print(f"Running Epoch {ep + 1}/{nb_epochs}")
        L = LossBox('4.5f')

        for i, ((x, y), (frnk_ka, frnk_kc, frnk_y)) in enumerate(zip(train_loader, frnk_loader)):
            x, y = x.to(model.device), y.to(model.device)
            #a = A[y]
            #a_mask = A_mask[y]  # Masking with continuous attributes seems to work better!

            real = torch.FloatTensor(x.shape[0], 1).fill_(1.0).to(model.device)
            lreal = torch.FloatTensor(x.shape[0], 1).fill_(.9).to(model.device)
            lfake = torch.FloatTensor(x.shape[0], 1).fill_(0.1).to(model.device)

            # -----------------
            #  Train Generator
            # -----------------
            g_opt.zero_grad()
            frnk_ka, frnk_kc, frnk_y = frnk_ka.to(model.device), frnk_kc.to(model.device), frnk_y.to(model.device)
            frnk_mask = A_mask[frnk_y]
            frnk_x = model.autoencoder.decode(frnk_ka, frnk_kc, frnk_mask)
            pred = model.discriminator(frnk_x)
            attr_pred = model.attr_predictor(frnk_x)
            cls_pred = model.classifier(frnk_x)
            g_adv =  adversarial_loss(pred, real)
            g_attr = auxiliary_attr_loss(attr_pred, A[frnk_y])
            g_cls = auxiliary_cls_loss(cls_pred, frnk_y)
            g_loss = ( g_adv + g_attr + g_cls) / 3
            g_loss.backward()
            g_opt.step()


            # ---------------------
            #  Train Discriminator
            # ---------------------
            d_opt.zero_grad()
            x1 = model.autoencoder(x, A[y])

            # Real image loss
            pred_real = model.discriminator(x1)
            attr_pred_real = model.attr_predictor(x1)
            cls_pred_real = model.classifier(x1)
            d_real_adv = adversarial_loss(pred_real, lreal)
            d_real_attr = auxiliary_attr_loss(attr_pred_real, A[y])
            d_real_cls = auxiliary_cls_loss(cls_pred_real, y)
            d_real_loss = (d_real_adv + d_real_attr + d_real_cls*0.01) / 2

            # Fake image loss
            frnk_x = frnk_x.detach()
            pred_fake = model.discriminator(frnk_x)
            attr_pred_fake = model.attr_predictor(frnk_x)
            cls_pred_fake = model.classifier(frnk_x)
            d_fake_adv = adversarial_loss(pred_fake, lfake)
            d_fake_attr = auxiliary_attr_loss(attr_pred_fake, A[frnk_y])
            d_fake_cls = auxiliary_cls_loss(cls_pred_fake, frnk_y)
            d_fake_loss = (d_fake_adv + d_fake_attr + d_fake_cls*0.01) / 2

            # Reconstruction loss
            rec_loss = reconstruction_loss(x1, x)

            # Total discriminator loss
            d_loss = rec_loss + (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_opt.step()

            if verbose >= 3:
                zero = torch.tensor([0.])
                L.append('d_real_adv', d_real_adv);
                L.append('d_fake_adv', d_fake_adv);
                L.append('g_adv', g_adv);
                L.append('-', zero)
                L.append('d_real_attr', d_real_attr);
                L.append('d_fake_attr', d_fake_attr);
                L.append('g_attr', g_attr);
                L.append('--', zero)
                L.append('d_real_cls', d_real_cls);
                L.append('d_fake_cls', d_fake_cls);
                L.append('g_cls', g_cls);
                L.append('---', zero)
                L.append('rec_loss', rec_loss)
                L.append('----', zero)
                L.append('d_real_loss', d_real_loss);
                L.append('d_fake_loss', d_fake_loss);
                L.append('d_loss', d_loss);
                L.append('g_loss', g_loss)





        if verbose >= 3:
            print(L)

        # train_accs.append(run_test(netA, train, device=device, verbose=verbose>=2))
        # if val is not None:
        #     valid_accs.append(gen_zsl_test(model, val, device=device, verbose=verbose >= 2))
        if test_period is not None and (ep + 1) % test_period == 0:
            test_classifier = get_fc_net(feats_dim, hidden_sizes=None, output_size=nb_test_classes, hidden_activations=nn.ReLU).to(model.device)
            acc = gen_zsl_test(model.autoencoder, test_classifier, train, test_unseen, nb_gen_class_samples=test_gen_samples,
                               use_masking=test_use_masking,
                               infinite_dataset=infinite_dataset,
                               adapt_epochs=test_epochs, adapt_lr=test_lr, device=model.device)
            test_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                print(f'New best accuracy: {best_acc:2.4f}')
                model.save(state_dir, epoch=None, acc=acc)

            else:
                print(f'Current accuracy: {acc:2.4f}')
                print(f'Old best accuracy: {best_acc:2.4f}')
            model.save(state_dir, ep, acc)

        if patience is not None:
            pscore = valid_accs[-1] if val is not None else test_accs[-1]
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


def gen_zsl_test(autoencoder: Autoencoder,
                 test_classifier: nn.Module,
                 train_dict,
                 zsl_unseen_test_dict,
                 nb_gen_class_samples=100,
                 adapt_epochs: int = 5,
                 adapt_lr: float = .0001,
                 adapt_bs: int = 128,
                 attrs_key='class_attr',
                 attrs_mask_key='class_attr',  # 'class_attr_bin'
                 use_masking=False,
                 infinite_dataset=True,
                 device=None):
    feats_dim = train_dict['feats'].shape[1]
    nb_new_classes = len(zsl_unseen_test_dict['class_attr_bin'])
    A = torch.tensor(zsl_unseen_test_dict[attrs_key]).float().to(device)
    A_mask = torch.tensor(zsl_unseen_test_dict[attrs_mask_key]).float().to(device)

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
                                  train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                  zsl_unseen_test_dict['class_attr_bin'], device=device)
    else:
        dataset = FrankensteinDataset(nb_gen_class_samples, enc_fn,
                                      train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                      zsl_unseen_test_dict['class_attr_bin'], device=device)

    data_loader = DataLoader(dataset, batch_size=adapt_bs, num_workers=0, shuffle=True)

    ######## TRAINING NEW CLASSIFIER ###########
    optim = torch.optim.Adam(test_classifier.parameters(), lr=adapt_lr)
    best_unseen_acc = 0
    best_classifier_state = None
    for ep in range(adapt_epochs):
        preds = []
        y_trues = []
        losses = []
        for data in data_loader:
            ka, kc, Y = data[0].to(device), data[1].to(device), data[2].to(device)
            optim.zero_grad()
            a_mask = A_mask[Y] if use_masking else None
            X = autoencoder.decode(ka, kc, a_mask)
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


# %%


def tsne(model: AttrGanModel, test_dicts, train_dict, nb_pca=None, nb_gen_class_samples=200, infinite_dataset=False,
         target='feats', # 'attr_emb', 'cntx_emb'
         append_title='',
         attrs_key='class_attr', mask_attrs_key='class_attr',
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
            frankenstain_dataset = InfiniteDataset(nb_gen_class_samples*nb_classes, model.autoencoder.encode,
                                                   train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                                   data_dict['class_attr_bin'], device=model.device)
        else:
            frankenstain_dataset = FrankensteinDataset(nb_gen_class_samples, model.autoencoder.encode,
                                                       train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                                       data_dict['class_attr_bin'], device=model.device)
        frankenstain_loader = DataLoader(frankenstain_dataset, batch_size=128, num_workers=0, shuffle=True)


        if target in ['cntx_emb', 'attr_emb']:
            KA1 = []; KC1 = []; Y1 = []
            for ka, kc, y in frankenstain_loader:
                ka = model.autoencoder.mask_ka(ka.to(model.device), test_A_mask[y.to(model.device)])
                KA1.append(ka);  KC1.append(kc); Y1.append(y)

            KA = []; KC = []; Y = []
            for x, y in data_loader:
                ka, kc = model.autoencoder.encode(x.to(model.device))
                ka = model.autoencoder.mask_ka(ka, test_A_mask[y.to(model.device)])
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
                        palette=sns.color_palette("hls", nb_classes), legend=False, alpha=0.4, ax=axs[i])
        sns.scatterplot(x=embeddings1[:,0], y=embeddings1[:, 1], hue=Y1,
                        palette=sns.color_palette("hls", nb_classes), legend=False, alpha=1, ax=axs[i])
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, title+'.png'))
    else:
        plt.show()


#%%

def build_attrgan_A(feats_dim: int, nb_attributes: int, nb_train_classes: int, nb_test_classes:int, device: Union[str, None]=None) -> AttrGanModel:
    return AttrGanModel(feats_dim, nb_attributes, nb_train_classes, ka_dim=32, kc_dim=256,  z_dim=None,
                        autoencoder_hiddens=AutoencoderHiddens(),
                        attribute_predictor_hiddens=(512,),
                        discriminator_hiddens=(512,),
                        device=device)


build_attrgan_fn = Callable[[int, int, int, int, Union[str, None]], AttrGanModel]

def init_exp(get_model_fn: build_attrgan_fn, gpu, seed, use_valid=False, download_data=False, preprocess_data=False, dataset='AWA2'):
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



import shutil


def run_train(epochs, exp_name, model, train, test_unseen, test_seen, val):
    state_dir = f'states/{exp_name}'
    try:
        os.makedirs(state_dir, exist_ok=False)
    except FileExistsError as e:
        print('\n\nSTATE FOLDER WITH THIS EXP_NAME ALREADY EXISTS!!')
        raise e
    shutil.copy('exp_6_autogan.py', os.path.join(state_dir, f'_exp_6__{exp_name}.py'))


    lr = .00001

    _, _, test_accs = exp(model, train, test_unseen, test_seen, val,
                          use_masking=True,
                          attrs_key='class_attr', mask_attrs_key='class_attr',
                          #a_lr=.000002, c_lr=.0001,
                          g_lr=lr, d_lr=lr, c_lr=.0001,
                          infinite_dataset=True,  # False
                          pretrain_cls_epochs=10,
                          test_lr=.0004, test_gen_samples=200,  # 2000, 800
                          test_period=1, test_epochs=6,
                          test_use_masking=True,
                          nb_epochs=epochs,
                          verbose=3,
                          state_dir=f'states/{exp_name}')


def run_tsne(epochs, exp_name, model, train, test_unseen, test_seen, val):
    state_dir = f'states/{exp_name}'
    tsne_dir = f'tsne/{exp_name}'
    for i in range(0, epochs + 1, 1):
        model.load(state_dir, epoch=i)
        app_title = f'- ep={i} - acc={model.acc * 100:2.2f}'
        tsne(model, [test_unseen, train], train, infinite_dataset=True, append_title=app_title, savepath=tsne_dir, target='feats')
        tsne(model, [test_unseen, train], train, infinite_dataset=True, append_title=app_title, savepath=tsne_dir, target='attr_emb')


if __name__ == '__main__':
    epochs=60
    model, train, test_unseen, test_seen, val = init_exp(build_attrgan_A, gpu=0, seed=42, use_valid=False,
                                                         download_data=False, preprocess_data=False, dataset='AWA2')
    exp_name = 'exp_6__1'
    run_train(epochs, exp_name, model, train, test_unseen, test_seen, val )
    #run_tsne(epochs,exp_name, model, train, test_unseen, test_seen, val )


