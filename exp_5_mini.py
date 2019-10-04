from sklearn import metrics
from sortedcontainers import SortedSet
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader

import data
from disentangle.dataset_generator import InfiniteDataset, AsyncInfiniteDataset, FrankensteinDataset

from disentangle.layers import GradientReversal, BlockLinear, get_fc_net
from utils import init

from utils import NP
import torch.nn.functional as NN
from sklearn.decomposition import PCA
from tsnecuda import TSNE
import os

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


class Autoencoder(nn.Module):
    def __init__(self, feats_dim, ka_dim, kc_dim, nb_attributes):
        super().__init__()
        self.feats_dim = feats_dim
        self.ka_dim = ka_dim
        self.kc_dim = kc_dim
        self.nb_attributes = nb_attributes
        GRL = GradientReversal
        #
        self.encKA = nn.Sequential(nn.Linear(feats_dim, ka_dim * nb_attributes), nn.LeakyReLU(),
                                   #BlockLinear(ka_dim, ka_dim, nb_attributes), nn.LeakyReLU(),
                                   #nn.Linear(2048, ka_dim*nb_attributes), nn.LeakyReLU()
                                   )
        self.decA = nn.Sequential(#BlockLinear(ka_dim, ka_dim, nb_attributes), nn.LeakyReLU(),
                                  BlockLinear(ka_dim, 1, nb_attributes), nn.Sigmoid()
                                  )
        self.decAD = nn.Sequential(GRL(1),
                                   BlockLinear(ka_dim, nb_attributes-1, nb_attributes), nn.Sigmoid()  # nn.LeakyReLU(),
                                   #BlockLinear(ka_dim, 1, nb_attributes), nn.Sigmoid()
                                   )
        self.encKC = nn.Sequential(nn.Linear(feats_dim, kc_dim), nn.LeakyReLU(),
                                   # nn.Linear(2048, kc_dim), nn.LeakyReLU()
                                   )
        self.decAC = nn.Sequential(GRL(1),
                                   nn.Linear(kc_dim, nb_attributes), nn.Sigmoid(),  # nn.LeakyReLU(),
                                   #nn.Linear(kc_dim, nb_attributes), nn.Sigmoid(),
                                   )

        self.decX = nn.Sequential(nn.Linear(ka_dim * nb_attributes + kc_dim, 2048), nn.LeakyReLU(),
                                  nn.Linear(2048, feats_dim), nn.ReLU()
                                  )



    def apply_mask(self, ka, a_mask):
        return ka * a_mask.repeat_interleave(self.ka_dim, dim=-1)

    def decode_a_dis(self, ka):
        return self.decAD(ka).view(-1, self.nb_attributes, self.nb_attributes - 1)

    def encode_x(self, x, mask=None):
        ka = self.encKA(x) if mask is None else self.apply_mask(self.encKA(x), mask)
        kc = self.encKC(x)
        return ka, kc

    def decode_x(self, ka, kc):
        return self.decX(torch.cat([ka, kc], dim=1))

    def forward(self, x, mask=None):
        ka, kc = self.encode_x(x, mask)
        a1 = self.decA(ka)
        ad = self.decode_a_dis(ka)
        ac = self.decAC(kc)
        x1 = self.decode_x(ka, kc)
        return x1, a1, ac, ad


def diag_nondiag_idx(rows_cols):
    diag_idx = SortedSet([(i, i) for i in range(rows_cols)])
    all_idx = SortedSet([tuple(l) for l in torch.triu_indices(rows_cols, rows_cols, -rows_cols).transpose(0, 1).numpy().tolist()])
    non_diag_idx = all_idx.difference(diag_idx)
    non_diag_idx = torch.Tensor(non_diag_idx).transpose(0, 1).long()
    return diag_idx, non_diag_idx




def exp(autoencoder, train, test,
        bin_attr=False, bin_mask=False,
        bs=128, a_lr=.00001, nb_epochs=100,
        train_with_frankenstain=True, test_tsne=False,
        pretrain_cls_epochs=10, c_lr=.0002,
        test_period=3, adapt_gen_samples=500, adapt_epochs=10, adapt_lr=.0001,
        device=None):
    attrs_key = 'class_attr_bin' if bin_attr else 'class_attr'
    mask_attrs_key =  'class_attr_bin' if bin_mask else 'class_attr'

    train_dataset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    nb_train_classes, nb_attributes = train[attrs_key].shape
    nb_test_classes, _ = test[attrs_key].shape
    A = torch.tensor(train[attrs_key]).float().to(device)
    A_mask = torch.tensor(train[mask_attrs_key]).float().to(device)


    diag_idx, non_diag_idx = diag_nondiag_idx(nb_attributes)
    #autoencoder_params = list(model.autoencoder.parameters())
    flatten = lambda l: [item for sublist in l for item in sublist]
    # decoder_params = flatten([list(m.parameters()) for m in [autoencoder.decK_X, autoencoder.decKA, autoencoder.decKC]])
    # encoder_params = flatten([list(m.parameters()) for m in [autoencoder.encX_KA, autoencoder.encX_KC]])

    classifier = Classifier(autoencoder.feats_dim, nb_train_classes, hidden_units=None).to(device)
    if pretrain_cls_epochs > 0:
        print("Start classifier pre-training...")
        c_opt = torch.optim.Adam(classifier.parameters(), lr=c_lr)
        for i in range(pretrain_cls_epochs):
            L = LossBox('4.5f')
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                c_opt.zero_grad()
                logits = classifier(x)
                l_cls = NN.cross_entropy(logits, y)
                l_cls.backward()
                c_opt.step()
                L.append('cls_x', l_cls);
            print(L)

    # *********** AUTOENCODER TRAINING
    print("Start autoencoder training...")
    opt = torch.optim.Adam(list(autoencoder.parameters()) +
                           list(classifier.parameters()), lr=a_lr)
    best_acc=0
    l_nll1 = torch.tensor([.0]).to(device)
    for ep in range(nb_epochs):
        print("\n")
        print(f"Running Epoch {ep + 1}/{nb_epochs}")
        L = LossBox('4.5f')
        fL = LossBox('4.5f')

        # frnk_dataset = AsyncInfiniteDataset(len(train_dataset), autoencoder.encode_x,
        #                                     train['feats'], train['labels'], train['attr_bin'], train['class_attr_bin'],
        #                                     ka_dim=autoencoder.ka_dim, kc_dim=autoencoder.kc_dim, device=device)
        frnk_dataset = InfiniteDataset(len(train_dataset), autoencoder.encode_x,
                                       train['feats'], train['labels'], train['attr_bin'], train['class_attr_bin'],
                                       device=device)

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            a = A[y] # NB: Masking with continuous attributes seems to work better!
            a_mat_true = a[:, None, :].repeat(1, nb_attributes, 1)
            a_non_diag = a_mat_true[:, non_diag_idx[0], non_diag_idx[1]].reshape(a.shape[0], nb_attributes, nb_attributes - 1)
            #a_non_diag = a_non_diag.transpose(1, 2)

            opt.zero_grad()
            x1, a1, ac, ad = autoencoder.forward(x, a)
            l_rec_x = NN.mse_loss(x1, x, reduction='none').sum(dim=-1).mean()  # image reconstruction
            if bin_attr:
                l_rec_a1 = NN.binary_cross_entropy(a1, a)  # attribute reconstruction
                l_rec_ac = NN.binary_cross_entropy(ac, a)  # context disentanglement
                l_rec_ad = NN.binary_cross_entropy(ad, a_non_diag)  # attr disentanglement
            else:
                l_rec_a1 = NN.mse_loss(a1, a, reduction='none').sum(dim=-1).mean()  # attribute reconstruction
                l_rec_ac = NN.mse_loss(ac, a, reduction='none').sum(dim=-1).mean()  # context disentanglement
                l_rec_ad = NN.mse_loss(ad, a_non_diag, reduction='none').sum(dim=-1).mean()  # attr disentanglement

            L.append('x', l_rec_x);
            L.append('a1', l_rec_a1);
            L.append('ac', l_rec_ac);
            L.append('ad', l_rec_ad);
            if classifier is not None:
                logits = classifier(x)
                logits1 = classifier(x1)
                l_nll = NN.cross_entropy(logits, y)
                l_nll1 = NN.cross_entropy(logits1, y)
                L.append("NLL", l_nll)
                L.append("NLL1", l_nll1)


            l = l_rec_x*1  + l_rec_a1*85 + l_rec_ac*85 + l_rec_ad*0# + l_nll*.1 + l_nll1*1

            if train_with_frankenstain:
                frnk_ka, frnk_kc, frnk_y = [t.to(device) for t in  frnk_dataset.get_items_with_class(y)]
                frnk_x1 = autoencoder.decode_x(frnk_ka, frnk_kc)
                frnk_a1 = autoencoder.decA(frnk_ka)
                frnk_ad = autoencoder.decode_a_dis(frnk_ka)
                frnk_ac = autoencoder.decAC(frnk_kc)
                l_rec_x = NN.mse_loss(frnk_x1, x, reduction='none').sum(dim=-1).mean()  # image reconstruction
                if bin_attr:
                    l_rec_a1 = NN.binary_cross_entropy(frnk_a1, a)  # attribute reconstruction
                    l_rec_ac = NN.binary_cross_entropy(frnk_ac, a, )  # context disentanglement
                    l_rec_ad = NN.binary_cross_entropy(frnk_ad, a_non_diag)  # attr disentanglement
                else:
                    l_rec_a1 = NN.mse_loss(frnk_a1, a, reduction='none').sum(dim=-1).mean()  # attribute reconstruction
                    l_rec_ac = NN.mse_loss(frnk_ac, a, reduction='none').sum(dim=-1).mean()  # context disentanglement
                    l_rec_ad = NN.mse_loss(frnk_ad, a_non_diag, reduction='none').sum(dim=-1).mean()  # attr disentanglement

                fL.append('x', l_rec_x);
                fL.append('a1', l_rec_a1);
                fL.append('ac', l_rec_ac);
                fL.append('ad', l_rec_ad);
                if classifier is not None:
                    logits1 = classifier(frnk_x1)
                    l_nll1 = NN.cross_entropy(logits1, y)
                    fL.append('NLL1', l_nll1);

                fl = l_rec_x*5  + l_rec_a1*85 + l_rec_ac*85 + l_rec_ad*0 #+ l_nll1*1
                (fl+l).backward()
            else:
                l.backward()

            opt.step()

        print(L)
        print("=== Frankenstain stuff ===")
        print(fL)

        if test_period is not None and (ep + 1) % test_period == 0:
            if test_tsne:
                tsne(autoencoder, [test, train], train, nb_gen_class_samples=adapt_gen_samples, infinite_dataset=True, target='feats',
                     attrs_key=attrs_key, mask_attrs_key=mask_attrs_key, device=device)

            test_classifier=Classifier(autoencoder.feats_dim, nb_test_classes, hidden_units=(2048,))
            acc = gen_zsl_test(autoencoder, test_classifier, train, test, nb_gen_class_samples=adapt_gen_samples,
                               attrs_key=attrs_key, attrs_mask_key=mask_attrs_key,
                               encode_during_test=True, use_masking=True,
                               infinite_dataset=True,
                               adapt_epochs=adapt_epochs, adapt_lr=adapt_lr, device=device)

            #.append(acc)
            if acc > best_acc:
                best_acc = acc
                print(f'New best accuracy: {best_acc:2.4f}')
                #model.save(state_dir, epoch=None, acc=acc)
            else:
                print(f'Current accuracy: {acc:2.4f}')
                print(f'Old best accuracy: {best_acc:2.4f}')
           # model.save(state_dir, ep, acc)



    return



#%% TEST
import numpy as np
def gen_zsl_test(autoencoder: Autoencoder,
                 test_classifier: Classifier,
                 train_dict,
                 zsl_unseen_test_dict,
                 nb_gen_class_samples=100,
                 adapt_epochs: int = 5,
                 adapt_lr: float = .0001,
                 adapt_bs: int = 128,
                 attrs_key='class_attr_bin',
                 attrs_mask_key='class_attr_bin',  # 'class_attr_bin'
                 encode_during_test=False,
                 use_masking=False,
                 infinite_dataset=True,
                 device=None):
    feats_dim = train_dict['feats'].shape[1]
    nb_new_classes = len(zsl_unseen_test_dict['class_attr_bin'])
    A = torch.tensor(zsl_unseen_test_dict[attrs_key]).float().to(device)
    A_mask = torch.tensor(zsl_unseen_test_dict[attrs_mask_key]).float().to(device)

    ######## PREPARE NEW CLASSIFIER ###########
    test_classifier.reset_linear_classifier(nb_new_classes).to(device)

    def test_on_test():
        unseen_test_feats = torch.tensor(zsl_unseen_test_dict['feats']).float()
        unseen_test_labels = torch.tensor(zsl_unseen_test_dict['labels']).long()
        dloader = DataLoader(TensorDataset(unseen_test_feats, unseen_test_labels), batch_size=adapt_bs, num_workers=2)
        losses, preds, y_trues = [], [], []
        for X, Y in dloader:
            X = X.to(device);
            Y = Y.to(device)
            # z, ka, kc, z1, x1, a1, ad, ac = net(X)
            if encode_during_test:
                ka, kc = autoencoder.encode_x(X)
                x1 = autoencoder.decode_x(ka, kc)
                logit = test_classifier(x1)
            else:
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
        return autoencoder.encKA(X), autoencoder.encKC(X)

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
            X = autoencoder.decode_x(ka, kc)
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


#%% TSNE

def tsne(model: Autoencoder, test_dicts, train_dict, nb_pca=None, nb_gen_class_samples=200, infinite_dataset=False,
         target='feats', # 'attr_emb', 'cntx_emb'
         append_title='',
         attrs_key='class_attr', mask_attrs_key='class_attr',
         savepath=None, dpi=250,
         device=None):
    test_dicts = [test_dicts] if isinstance(test_dicts, dict) else test_dicts

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(nrows=len(test_dicts), figsize=(20, 8 * len(test_dicts)), dpi=dpi)
    if len(test_dicts) is 1:
        axs = [axs]

    title = f'tsne {target}{append_title}'
    fig.suptitle(title, fontsize=16)

    for i, data_dict in enumerate(test_dicts):
        nb_classes = len(set(data_dict['labels']))
        test_A = torch.tensor(data_dict[attrs_key]).float().to(device)
        test_A_mask = torch.tensor(data_dict[mask_attrs_key]).float().to(device)
        X = torch.tensor(data_dict['feats']).float()
        Y = torch.tensor(data_dict['labels']).long()
        dataset = TensorDataset(X, Y)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

        if infinite_dataset:
            frankenstain_dataset = InfiniteDataset(nb_gen_class_samples * nb_classes, model.encode_x,
                                                   train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                                   data_dict['class_attr_bin'], device=device)
        else:
            frankenstain_dataset = FrankensteinDataset(nb_gen_class_samples, model.encode_x,
                                                       train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                                       data_dict['class_attr_bin'], device=device)
        frankenstain_loader = DataLoader(frankenstain_dataset, batch_size=128, num_workers=0, shuffle=True)


        if target in ['cntx_emb', 'attr_emb']:
            KA1 = []; KC1 = []; Y1 = []
            for ka, kc, y in frankenstain_loader:
                #ka = model.apply_mask(ka.to(device), test_A_mask[y.to(device)])
                KA1.append(ka);  KC1.append(kc); Y1.append(y)

            KA = []; KC = []; Y = []
            for x, y in data_loader:
                ka, kc = model.encode_x(x.to(device))
                #ka = model.apply_mask(ka, test_A_mask[y.to(device)])
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
                ka, kc, y = data[0].to(device), data[1].to(device), data[2].to(device)
                x1 = model.decode_x(ka, kc)
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


# %%

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
    feats_dim = len(train['feats'][0])


    ### INIT MODEL
    nb_train_classes, nb_attributes = train['class_attr'].shape
    nb_test_classes, _ = test_unseen['class_attr'].shape
    autoencoder = Autoencoder(feats_dim=feats_dim, ka_dim=32, kc_dim=256, nb_attributes=nb_attributes).to(device)
    return autoencoder, train, test_unseen, test_seen, val, device


def main():
    autoencoder, train, test_unseen, test_seen, val, device =\
        init_exp(gpu=0, seed=42)

    exp(autoencoder, train, test_unseen,
        bin_attr=True, bin_mask=True,
        bs=128, a_lr=.0001, nb_epochs=50,
        train_with_frankenstain=True,
        pretrain_cls_epochs=0,  # 10,

        test_period=1, test_tsne=False,
        adapt_gen_samples=200, adapt_epochs=10,

        device=device)


# %%
if __name__ == '__main__':
    main()
