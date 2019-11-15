import torch
import numpy as np
from torch import nn

from disentangle.layers import L2Norm, BlockLinear
from utils import NP

from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Callable, Union, List, Tuple
import data

from disentangle.loss_box import LossBox
from utils import init
from sklearn import metrics

import torch.nn.functional as NN
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
from disentangle.model import Model

class CosineLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__(in_features, out_features, bias=False)

    def forward(self, x):
        x_norm = x.norm(p=2, dim=1)[:, None]
        w_norm = self.weight.norm(p=2, dim=1)[:, None]
        #self.weight.data /= w_norm
        return F.linear(x/x_norm, self.weight/w_norm)
        #return F.linear(x, self.weight)/x_norm.matmul(w_norm.transpose(0, 1))


class Encoder(Model):
    def __init__(self, nb_attr, in_feats=2048, hidden_dim=2048, latent_attr_dim=64):
        super().__init__()
        self.nb_attr = nb_attr
        self.in_feats = in_feats
        self.hidden_dim = 2048
        self.latent_attr_dim = 2048
        self.net = nn.Sequential(nn.Linear(in_feats, hidden_dim), nn.ReLU(),
                                 #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, latent_attr_dim * nb_attr))
    def forward(self, x):
        return self.net(x)

class AttrDecoder(Model):
    def __init__(self, nb_attr, hidden_dim=2048, latent_attr_dim=64):
        super().__init__()
        self.nb_attr = nb_attr
        self.hidden_dim = 2048
        self.latent_attr_dim = 2048
        self.net = nn.Sequential(#nn.Linear(latent_attr_dim * nb_attr, hidden_dim), nn.ReLU(),
                                 #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, nb_attr))
    def forward(self, x):
        return self.net(x)



#%% TEST

#%% MAIN
from torch import optim
def to(tensors, *args, **kwargs):
    return (t.to(*args, **kwargs) for t in tensors)

if __name__ == '__main__':
    nb_epochs = 60
    ATTR_IN_01=True
    #device, train, test_unseen, test_seen, val = init_exp(gpu=0,seed=None, dataset='AWA2', attr_in_01=ATTR_IN_01)
    device, train, test_unseen, test_seen, val = init_exp(gpu=0, seed=42, dataset='AWA2', attr_in_01=ATTR_IN_01)
    test_period = 2
    lr = .001
    wd = 0
    momentum=.7
    latent_attr_dim=4

    # MANAGE DATA
    nb_train_examples, feats_dim = train['feats'].shape
    nb_train_classes, nb_attributes = train['class_attr'].shape
    nb_test_classes, _ = test_unseen['class_attr'].shape
    A_mask = torch.tensor(train['class_attr_bin']).float()
    A = torch.tensor(train['class_attr']).float()
    train_attribute_masks = torch.tensor(train['attr_bin']).float()
    train_attributes = torch.tensor(train['attr']).float()
    train_features = torch.tensor(train['feats']).float()
    train_labels = torch.tensor(train['labels']).long()
    trainset = TensorDataset(train_features, train_labels, train_attribute_masks, train_attributes)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)


    E = Encoder(nb_attributes, feats_dim, hidden_dim=2048, latent_attr_dim=latent_attr_dim).to(device)
    #AD = CosineLinear(nb_attributes*latent_attr_dim, nb_attributes).to(device)
    # AD = nn.Linear(nb_attributes*latent_attr_dim, nb_attributes).to(device)
    AD = BlockLinear(latent_attr_dim, 1, nb_attributes).to(device)
    C = CosineLinear(nb_attributes*latent_attr_dim, nb_train_classes).to(device)
    #C = nn.Linear(nb_attributes*latent_attr_dim, nb_train_classes).to(device)


    #opt = optim.SGD(list(E.parameters()) + list(C.parameters()), lr=lr, weight_decay=wd, momentum=momentum)
    opt = optim.Adam(list(E.parameters()) + list(C.parameters()), lr=lr, weight_decay=wd)


    def get_norm_W():
        return NP(C.weight / C.weight.norm(p=2, dim=1)[:, None])

    def get_avg_norm_feats():
        tloader = DataLoader(TensorDataset(train_features), batch_size=128, shuffle=False, drop_last=False)
        Z = []
        Y = train_labels.numpy()
        for x in tloader:
            Z.append(E(x[0].to(device)).detach().cpu())
        Z = torch.cat(Z)
        Z = Z/Z.norm(p=2, dim=1, keepdim=True)
        Z = Z.numpy()
        Z_perclass = []
        for y in set(Y):
            idx = np.argwhere(Y==y)[:,0]
            Z_perclass.append(np.mean(Z[idx], axis=0))
        return np.array(Z_perclass)

    for ep in range(nb_epochs):
        C.train(); E.train()
        L = LossBox()

        y_preds = []
        y_trues = []
        a_trues = []
        a_preds = []
        for i, (x, y, a_mask, a) in enumerate(train_loader):
            x, y, a_mask, a = to((x, y, a_mask, a), device)
            opt.zero_grad()
            z = E(x)
            logits = C(z)
            a1 = AD(z)
            sm =  torch.softmax(logits, dim=1)

            cce_loss = F.nll_loss(sm, y)  # seems to work better when using cosine
            #cce_loss = F.cross_entropy(logits, y)

            attr_loss = F.mse_loss(a1, a)#, reduction='none').mean(dim=0).sum()
            #attr_loss = F.mse_loss(a1, a)#, reduction='none').mean(dim=0).sum()

            (1 * cce_loss + 1*attr_loss).backward()
            opt.step()
            y_preds.append(logits.argmax(dim=1).detach().cpu())
            y_trues.append(y.detach().cpu())
            a_preds.append((a1>0.5).float().detach().cpu())
            a_trues.append(a_mask.detach().cpu())

            L.append('cce', cce_loss.cpu().detach())
            L.append('attr', attr_loss.cpu().detach())

        y_preds = torch.cat(y_preds)
        y_trues = torch.cat(y_trues)
        a_preds = torch.cat(a_preds)
        a_trues = torch.cat(a_trues)
        acc = (y_preds == y_trues).float().mean()
        attr_acc = (a_preds == a_trues).float().mean()

        print(f'====> Epoch: {ep+1}   -  train-acc={acc}  train-attr-acc-={attr_acc}')
        print(L)

        if (ep+1)%test_period == 0:
            exemplars = torch.tensor(get_avg_norm_feats())
            W = torch.tensor(get_norm_W())
            dist = torch.norm(exemplars-W, p=2, dim=1)
            print(f"Per class l2-distance (weight vs per-class-mean-embedding): {dist.mean()}")
            print(dist)

            # Test on unseen-test-set:
            # TODO: generate weight for classifier of unseen-classes
            # TODO: test on test-set
            pass

#%% TRASHCAN



