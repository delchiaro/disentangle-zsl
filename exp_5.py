from copy import deepcopy

from sortedcontainers import SortedSet
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

from data import get_dataset, preprocess_dataset, download_data, normalize_dataset
from disentangle.dataset_generator import InfiniteDataset, FrankensteinDataset

from disentangle.layers import get_fc_net, GradientReversal, GradientReversalFunction, get_1by1_conv1d_net, IdentityLayer
from utils import init

from sklearn import metrics
import numpy as np
from utils import NP
import torch.nn.functional as NN


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

class Autoencoder(nn.Module):
    def __init__(self,feats_dim, z_dim, ka_dim, kc_dim, nb_attributes):
        super().__init__()
        self.feats_dim = feats_dim
        self.z_dim = z_dim
        self.ka_dim = ka_dim
        self.kc_dim = kc_dim
        self.nb_attributes = nb_attributes
        GRL = GradientReversal
        self.encX_Z = get_fc_net(feats_dim, None, z_dim, nn.ReLU(), nn.ReLU())
        self.decZ_X = get_fc_net(z_dim, None, feats_dim, nn.ReLU(), nn.ReLU())
        #
        self.encZ_KA = get_fc_net(z_dim, None, ka_dim*nb_attributes, nn.ReLU(), nn.ReLU())
        self.encZ_KC = get_fc_net(z_dim, None, kc_dim, nn.ReLU(), nn.ReLU())
        self.decK_Z = get_fc_net(ka_dim*nb_attributes+kc_dim, None, z_dim, nn.ReLU(), nn.ReLU())
        #
        self.decKA_A = get_1by1_conv1d_net(ka_dim, None, 1, nn.ReLU(), nn.Sigmoid())
        self.decKA_disA = nn.Sequential(GRL(1), get_1by1_conv1d_net(ka_dim, None, nb_attributes-1, nn.ReLU(), nn.Sigmoid()))
        self.decKC_disC = nn.Sequential(GRL(1), get_fc_net(kc_dim, None, nb_attributes, nn.ReLU(), nn.Sigmoid()))


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

    def attr_decode(self, ka, a_mask=None):
        if a_mask is not None:
            ka = self.mask_ka(ka, a_mask)
        ka_conv = ka.view(ka.shape[0], self.ka_dim, self.nb_attributes)
        a1 = self.decKA_A(ka_conv).squeeze(dim=1)
        ad = self.decKA_disA(ka_conv).transpose(-1, -2)
        return a1, ad

    def cntx_attr_decode(self, kc):
        return self.decKC_disC(kc)

    def forward(self, x, a_mask=None):
        ka, kc = self.encode(x)  # Image Encode
        a1, ad = self.attr_decode(ka)   # Attribute decode and attribute disentangle decode
        #a1, ad = self.attr_decode(ka, a_mask)   # Attribute decode and attribute disentangle decode
        x1 = self.decode(ka, kc, a_mask)  # Image decode
        ac = self.cntx_attr_decode(kc)  # Context disentangle from attribute decode
        return ka, kc, x1, a1, ad, ac


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
    #label_exp = interlaced_repeat(label, nb_classes, dim=0)
    all_attrs_exp = all_mask_attrs.repeat([bs, 1])
    #
    decoded = autoencoder.decode(attr_enc_exp, cntx_enc_exp, all_attrs_exp)
    logits = classifier(decoded)
    t = torch.tensor([[t] for t in list(range(nb_classes)) * bs]).to(attr_enc.device)
    logits_diags = torch.gather(logits, 1, t).view(bs, nb_classes)
    return NN.cross_entropy(logits_diags, label)




def exp(seed=None, gpu=0, use_valid=False, use_masking=False, early_stop_patience=None,
        attrs_key='class_attr', mask_attrs_key='class_attr',
        a_lr=.00001, c_lr=.0001, pretrain_classifier_epochs=5,
        test_period=1, adapt_epochs=10, adapt_lr=.0001, adapt_gen_samples=300,
        encode_during_test=False, test_use_masking=True, infinite_dataset=True,
        verbose=2):
    ################### CONFIGURATIONS
    DOWNLOAD_DATA = False
    PREPROCESS_DATA = False
    DATASET = 'AWA2'  # 'CUB'
    device = init(gpu_index=gpu, seed=seed)
    if verbose >= 1:
        print('\n\n\n')
        print('*'*80)
        print(f'   Starting new exp with seed {seed}')
        print('*'*80)
        print()

    bs = 128
    nb_epochs = 200

     ################### START EXP
    if DOWNLOAD_DATA:
        download_data()
    if PREPROCESS_DATA:
        preprocess_dataset(DATASET)
    train, val, test_unseen, test_seen = get_dataset(DATASET, use_valid=use_valid, gzsl=False, mean_sub=False, std_norm=False, l2_norm=False)
    train, val, test_unseen, test_seen = normalize_dataset(train, val, test_unseen, test_seen, keys=('class_attr',), feats_range=(0, 1))
    feats_dim = len(train['feats'][0])
    nb_classes, nb_attributes = train[attrs_key].shape

    classes = sorted(set(train['labels'].tolist()))
    cls_counters = []
    for cls in classes:
        cls_counters.append(np.sum(train['labels'] == cls))
    cls_counters = torch.tensor(cls_counters).float().to(device)
    tot_examples = cls_counters.sum()


    ka_dim=32
    kc_dim=256  #best: 256, 512, 768
    z_dim = 2048
    autoencoder = Autoencoder(feats_dim, z_dim, ka_dim, kc_dim, nb_attributes).to(device)
    classifier = Classifier(feats_dim, nb_classes, (2048, ) ).to(device)
    cntx_classifier = nn.Sequential(GradientReversal(1), Classifier(kc_dim, nb_classes, )).to(device)

    train_dataset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    A = torch.tensor(train[attrs_key]).float().to(device)
    A_mask = torch.tensor(train[mask_attrs_key]).float().to(device)


    def diag_nondiag_idx(rows_cols):
        diag_idx = SortedSet([(i, i) for i in range(rows_cols)])
        all_idx = SortedSet([tuple(l) for l in torch.triu_indices(rows_cols, rows_cols, -rows_cols).transpose(0, 1).numpy().tolist()])
        non_diag_idx = all_idx.difference(diag_idx)
        non_diag_idx = torch.Tensor(non_diag_idx).transpose(0, 1).long()
        return diag_idx, non_diag_idx


    diag_idx, non_diag_idx = diag_nondiag_idx(nb_attributes)


    random_a_cntx_loss = torch.log(torch.tensor([nb_attributes]).float()).to(device)
    random_a_dis_loss = torch.log(torch.tensor([nb_attributes*(nb_attributes-1)]).float()).to(device)

    #run_test(netA, test_unseen, device=device)

    # *********** CLASSIFIER PRE-TRAINING
    c_opt = torch.optim.Adam(classifier.parameters(), lr=c_lr)
    for i in range(pretrain_classifier_epochs):
        L = LossBox('4.5f')
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            c_opt.zero_grad()
            logits = classifier(x)
            l_cls = NN.cross_entropy(logits, y)
            l_cls.backward()
            c_opt.step()
            L.append('cls_x', l_cls);
        if verbose >= 3:
            print(L)

    # *********** AUTOENCODER TRAINING
    a_opt = torch.optim.Adam(list(autoencoder.parameters()) + list(classifier.parameters()) + list(cntx_classifier.parameters()),
                             lr=a_lr)

    train_accs, valid_accs, test_accs = [], [], []
    best_patience_score = 0
    best_acc = 0
    patience = early_stop_patience
    for ep in range(nb_epochs):
        if verbose >=2:
            print("\n")
            print(f"Running Epoch {ep+1}/{nb_epochs}")
        L = LossBox('4.5f')
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            a = A[y]
            a_mask = A_mask[y]  # Masking with continuous attributes seems to work better!

            c_opt.zero_grad()
            logits = classifier(x)
            l_cls = NN.cross_entropy(logits, y)
            l_cls.backward()
            c_opt.step()


            a_opt.zero_grad()
            a_dis = a[:, None, :].repeat(1, nb_attributes, 1)
            a_non_diag = a_dis[:, non_diag_idx[0], non_diag_idx[1]].reshape(a.shape[0], nb_attributes, nb_attributes - 1)
            #a_diag = a_dis[:, diag_idx[0], diag_idx[1]].reshape(a.shape[0], nb_attributes)

            ka, kc, x1, a1, ad, ac = autoencoder.forward(x, a_mask if use_masking else None)

            l_rec_x = NN.mse_loss(x1, x) # image reconstruction
            l_rec_a1 = NN.l1_loss(a1, a) # attribute reconstruction
            l_rec_ad = NN.l1_loss(ad, a_non_diag) # attribute disentanglement
            l_rec_ac = NN.l1_loss(ac, a) # context disentanglement

            logits1 = classifier(x1)
            l_cls1 = NN.cross_entropy(logits1, y)

            logits = classifier(x)
            l_cls = NN.cross_entropy(logits, y)

            logits_cntx = cntx_classifier(kc)
            l_cls_cntx = NN.cross_entropy(logits_cntx, y)

            #l_cls_contrastive = disentangle_loss(autoencoder, classifier, ka, kc, y, A_mask)

            l =  l_rec_x * 5
            l += l_rec_a1 * 5
            l += l_rec_ad * 1
            l += l_rec_ac * 1

            l += l_cls * .01
            l += l_cls1 * .01
            l += l_cls_cntx * .001
            #l += l_cls_contrastive * .01

            l.backward()
            a_opt.step()

            if verbose >= 3:
                L.append('x', l_rec_x);
                L.append('a1', l_rec_a1); L.append('ad', l_rec_ad);  L.append('ac', l_rec_ac);
                L.append('cls_x', l_cls); L.append('cls_x1', l_cls1);
                L.append('l_cls_cntx', l_cls_cntx);
                #L.append('l_cls_contrastive', l_cls_contrastive);

        if verbose >= 3:
            print(L)

        #train_accs.append(run_test(netA, train, device=device, verbose=verbose>=2))
        # if val is not None:
        #     valid_accs.append(gen_zsl_test(model, val, device=device, verbose=verbose >= 2))
        if test_period is not None and (ep + 1) % test_period == 0:
            acc, adapt_state = gen_zsl_test(autoencoder, classifier, train, test_unseen, nb_gen_class_samples=adapt_gen_samples,
                                                 encode_during_test=encode_during_test, use_masking=test_use_masking,
                                                 infinite_dataset=infinite_dataset,
                                                 adapt_epochs=adapt_epochs, adapt_lr=adapt_lr, device=device)
            test_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                print(f'New best accuracy: {best_acc:2.4f}')
                torch.save({'autoencoder': autoencoder.state_dict(),
                            'classifier': classifier.state_dict(),
                            'cntx_classifier': cntx_classifier.state_dict(),
                            'test_classifier': adapt_state,
                            'adapt_acc': acc,
                            }, 'states/best_models.pt')
            else:
                print(f'Current accuracy: {acc:2.4f}')
                print(f'Old best accuracy: {best_acc:2.4f}')

            torch.save({'autoencoder': autoencoder.state_dict(),
                        'classifier': classifier.state_dict(),
                        'cntx_classifier': cntx_classifier.state_dict(),
                        'test_classifier': adapt_state,
                        'adapt_acc' : acc,
                        }, f'states/models_epoch_{ep+1:03d}.pt')


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
                 classifier: Classifier,
                 train_dict,
                 zsl_unseen_test_dict,
                 nb_gen_class_samples=100,
                 adapt_epochs: int = 5,
                 adapt_lr: float = .0001,
                 adapt_bs: int = 128,
                 attrs_key='class_attr',
                 attrs_mask_key='class_attr', #'class_attr_bin'
                 encode_during_test=False,
                 use_masking=False,
                 infinite_dataset=True,
                 device=None):
    feats_dim = train_dict['feats'].shape[1]
    nb_new_classes = len(zsl_unseen_test_dict['class_attr_bin'])
    A = torch.tensor(zsl_unseen_test_dict[attrs_key]).float().to(device)
    A_mask = torch.tensor(zsl_unseen_test_dict[attrs_mask_key]).float().to(device)

    ######## PREPARE NEW CLASSIFIER ###########
    classifier = deepcopy(classifier)
    classifier.reset_linear_classifier(nb_new_classes)

    def test_on_test():
        unseen_test_feats = torch.tensor(zsl_unseen_test_dict['feats']).float()
        unseen_test_labels = torch.tensor(zsl_unseen_test_dict['labels']).long()
        dloader = DataLoader(TensorDataset(unseen_test_feats, unseen_test_labels), batch_size=adapt_bs, num_workers=2)
        losses, preds, y_trues = [], [], []
        for X, Y in dloader:
            X = X.to(device); Y = Y.to(device)
            #z, ka, kc, z1, x1, a1, ad, ac = net(X)
            if encode_during_test:
                ka, kc = autoencoder.encode(X)
                a1 = autoencoder.attr_decode(ka) if use_masking else None
                x1 = autoencoder.decode(ka, kc, a1)
                logit = classifier(x1)
            else:
                logit = classifier(X)
            loss = NN.cross_entropy(logit, Y)
            losses.append(loss.detach().cpu()); preds.append(logit.argmax(dim=1)); y_trues.append(Y)

        preds = torch.cat(preds); y_trues = torch.cat(y_trues); unseen_loss = torch.stack(losses).mean()
        unseen_acc = np.mean([metrics.recall_score(NP(y_trues), NP(preds), labels=[k], average=None) for k in sorted(set(NP(y_trues)))])
        return unseen_loss, unseen_acc

    ######## DATA GENERATION/PREPARATION ###########
    def enc_fn(X):
        Z = autoencoder.encX_Z(X)
        return autoencoder.encZ_KA(Z), autoencoder.encZ_KC(Z)

    if infinite_dataset:
        dataset = InfiniteDataset(nb_gen_class_samples, enc_fn,
                                  train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                  zsl_unseen_test_dict['class_attr_bin'], device=device)
    else:
        dataset = FrankensteinDataset(nb_gen_class_samples, enc_fn,
                                  train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                  zsl_unseen_test_dict['class_attr_bin'], device=device)

    data_loader = DataLoader(dataset, batch_size=adapt_bs, num_workers=0, shuffle=True)


    ######## TRAINING NEW CLASSIFIER ###########
    optim = torch.optim.Adam(classifier.net[-1].parameters(), lr=adapt_lr)
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
            logit =  classifier(X)
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
        print(f"Classifier adaptation - Epoch {ep+1}/{adapt_epochs}:   Loss={classifier_losses:1.5f}    Acc={acc:1.4f}  -  uLoss={unseen_loss:1.5f}   uAcc={unseen_acc:1.5f}")
        if unseen_acc > best_unseen_acc:
            best_unseen_acc = unseen_acc
            best_classifier_state = classifier.state_dict()
    return best_unseen_acc, best_classifier_state



#%%
if __name__ == '__main__':
    # _, _, test_accs = exp(42, gpu=0, verbose=3, early_stop_patience=50, use_valid=False)
    _, _, test_accs = exp(42, gpu=0, verbose=3,
                          use_valid=False, use_masking=True,
                          attrs_key='class_attr', mask_attrs_key='class_attr',
                          a_lr=.000002, c_lr=.0001,
                          pretrain_classifier_epochs=0,
                          test_period=1, adapt_epochs=10,
                          adapt_lr=.0004, adapt_gen_samples=2000, #800
                          encode_during_test=False,
                          infinite_dataset=True, # False
                          test_use_masking=True)


    #%%
    #
    # seeds = [42, 55, 0, 123]
    # best_accs = {}
    # nones = 0
    # for seed in seeds:
    #     _, _, test_accs = exp(seed, gpu=0, verbose=2, early_stop_patience=5, use_valid=False, la=1, lc=1, lx=0, grl=100)
    #     ep = np.argmax(test_accs)
    #     if seed is None:
    #         seed = f'None-{nones}'
    #         nones+=1
    #     best_accs[seed] = (test_accs[ep], ep+1)
    #
    # print("Best test accuracies and epoch per each seed: ")
    # print(best_accs)
    #
    # print(f"Mean best score: {np.mean([A[0] for A in best_accs.values()])}")