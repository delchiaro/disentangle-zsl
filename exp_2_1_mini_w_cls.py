from copy import deepcopy

from sortedcontainers import SortedSet
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader

from data import get_dataset, preprocess_dataset, download_data, normalize_dataset
from disentangle.dataset_generator import InfiniteDataset, FrankensteinDataset

from disentangle.layers import get_fc_net, GradientReversal, get_1by1_conv1d_net
from disentangle.net import DisentangleEncoder
from disentangle.utils import interlaced_repeat
from utils import init

#%%

#%%
from sklearn import metrics
import numpy as np
from utils import NP



def run_test(net,
             train_dict, zsl_unseen_test_dict,
             nb_gen_class_samples=100,
             adapt_epochs: int = 5,
             adapt_lr: float = .0001,
             adapt_bs: int = 128,
             use_infinite_dataset=True,
             device=None):
    feats_dim = train_dict['feats'].shape[1]
    nb_new_classes = len(zsl_unseen_test_dict['class_attr_bin'])

    classifier = deepcopy(net.classifier)
    classifier[-1] = nn.Linear(net.classifier_feats_dim, nb_new_classes).to(device)
    ######## PREPARE NEW CLASSIFIER ###########
    optim = torch.optim.Adam(classifier.parameters(), lr=adapt_lr)

    def test_on_test():
        unseen_test_feats = torch.tensor(zsl_unseen_test_dict['feats']).float()
        unseen_test_labels = torch.tensor(zsl_unseen_test_dict['labels']).long()
        dloader = DataLoader(TensorDataset(unseen_test_feats, unseen_test_labels), batch_size=adapt_bs, num_workers=2)
        losses, preds, y_trues = [], [], []
        for X, Y in dloader:
            X = X.to(device); Y = Y.to(device)

            # # Decode without attributes:
            # attr_enc, cntx_enc = net.enc(X)
            # X1 = net.x_decoder(torch.cat([attr_enc, cntx_enc], dim=1))
            # logit = classifier(X1)
            #
            # Use original img-feats:
            z = net.enc.pre_encoder(X)
            logit = classifier(z)

            # # Decode with ALL attributes and take diagonal logit:
            # logit = net.classify_unseen_enc(classifier, X, torch.tensor(zsl_unseen_test_dict['class_attr_bin']).to(device), device)


            loss = NN.cross_entropy(logit, Y)
            losses.append(loss.detach().cpu()); preds.append(logit.argmax(dim=1)); y_trues.append(Y)

        preds = torch.cat(preds); y_trues = torch.cat(y_trues); unseen_loss = torch.stack(losses).mean()
        unseen_acc = np.mean([metrics.recall_score(NP(y_trues), NP(preds), labels=[k], average=None) for k in sorted(set(NP(y_trues)))])
        return unseen_loss, unseen_acc

    ######## DATA GENERATION/PREPARATION ###########
    if use_infinite_dataset:
        dataset = InfiniteDataset(nb_gen_class_samples * nb_new_classes, net.enc,
                                  train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
                                  zsl_unseen_test_dict['class_attr_bin'], device=device)
    else:
        dataset = FrankensteinDataset(nb_gen_class_samples, net.enc,
                                      train_dict['feats'], train_dict['labels'], train_dict['class_attr_bin'],
                                      zsl_unseen_test_dict['class_attr_bin'], device=device)
    data_loader = DataLoader(dataset, batch_size=adapt_bs, num_workers=2, shuffle=True)



    ######## TRAINING NEW CLASSIFIER ###########
    for ep in range(adapt_epochs):
        preds = []
        y_trues = []
        losses = []
        for data in data_loader:
            attr_enc, cntx_enc, Y = data[0].to(device), data[1].to(device), data[2].to(device)
            optim.zero_grad()
            Z1 = net.z_decoder(torch.cat([attr_enc, cntx_enc], dim=1))
            logit = classifier(Z1)
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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        in_features = 2048
        pre_enc_units = (2048, 1024, 512)
        attr_enc_units = (128, 32)
        cntx_enc_units = (128, 128)
        attr_dec_hiddens = (32,)
        cntx_dec_hiddens = (128,)
        z_dec_hiddens = (2048, 2048,)
        z_dim = pre_enc_units[-1]
        attr_enc_dim = attr_enc_units[-1] * nb_attributes
        cntx_enc_dim = cntx_enc_units[-1]
        full_enc_dim = attr_enc_dim + cntx_enc_dim

        self.classifier_feats_dim = 512
        self.enc = DisentangleEncoder(in_features, nb_attributes, pre_enc_units, attr_enc_units, cntx_enc_units)
        self.attr_decoder = get_1by1_conv1d_net(attr_enc_units[-1], attr_dec_hiddens, 1, out_activation=nn.Sigmoid())
        self.dis_attr_decoder = nn.Sequential(GradientReversal(1),
                                              get_1by1_conv1d_net(attr_enc_units[-1], attr_dec_hiddens, nb_attributes-1,
                                                                  out_activation=nn.Sigmoid()))
        self.cntx_decoder = nn.Sequential(GradientReversal(1),
                                          get_fc_net(cntx_enc_dim, cntx_dec_hiddens, nb_attributes, out_activation=nn.Sigmoid()))
        self.z_decoder = get_fc_net(full_enc_dim, z_dec_hiddens, z_dim, out_activation=nn.ReLU())
        self.classifier = get_fc_net(pre_enc_units[-1], (1024, self.classifier_feats_dim,), nb_classes, hidden_activations=nn.LeakyReLU())


    def forward(self, x, a_bin=None):
        bs = x.shape[0]
        z = self.enc.pre_encoder(x)
        attr_enc = self.enc.attr_encoder(z)
        cntx_enc = self.enc.cntx_encoder(z)
        if a_bin is not None:
            mask = interlaced_repeat(a_bin, dim=1, times=self.enc.attr_enc_dim)
            attr_enc = attr_enc*mask
        attr_enc_tensor = attr_enc.view(bs, nb_attributes, -1).transpose(1, 2)
        a1 = self.attr_decoder(attr_enc_tensor).view(bs, -1)
        a1_dis = self.dis_attr_decoder(attr_enc_tensor).transpose(1, 2)
        a1_ctx = self.cntx_decoder(cntx_enc)
        full_enc = torch.cat([attr_enc, cntx_enc], dim=1)
        z1 = self.z_decoder(full_enc)
        logits1 = net.classifier(z1)
        logits = net.classifier(z)

        return logits, logits1, z, z1, a1, a1_dis, a1_ctx

    def classify_unseen_enc(self, classifier, x, all_attrs_bin, device=None):
        nb_classes = len(all_attrs_bin)
        bs = x.shape[0]

        attr_enc, cntx_enc = self.enc(x)
        attr_enc_exp = interlaced_repeat(attr_enc, dim=0, times=nb_classes)
        cntx_enc_exp = interlaced_repeat(cntx_enc, dim=0, times=nb_classes)

        all_attrs_exp = all_attrs_bin.repeat([bs, 1])
        mask = interlaced_repeat(all_attrs_exp, dim=1, times=self.enc.attr_enc_dim)
        all_attr_exp_maksed = attr_enc_exp * mask.float()

        full_enc = torch.cat([all_attr_exp_maksed, cntx_enc_exp], dim=1)
        decoded = self.z_decoder(full_enc)
        logits = classifier(decoded)
        t = torch.tensor([[t] for t in list(range(nb_classes)) * bs]).to(device)
        logits_diags = torch.gather(logits, 1, t).view(bs, nb_classes)
        return logits_diags

#%%

################### CONFIGURATIONS
DOWNLOAD_DATA = False
PREPROCESS_DATA = False
DATASET = 'CUB'  # 'CUB'
ATTRS_KEY = 'class_attr' # 'class_attr_bin'
#ATTRS_KEY = 'class_attr_bin' # 'class_attr_bin'
device = init(gpu_index=0, seed=42)
bs = 128
nb_epochs = 200

 ################### START EXP
if DOWNLOAD_DATA:
    download_data()
if PREPROCESS_DATA:
    preprocess_dataset(DATASET)
train, val, test_unseen, test_seen = get_dataset(DATASET, use_valid=False, gzsl=False, mean_sub=False, std_norm=False, l2_norm=False)
train, val, test_unseen, test_seen = normalize_dataset(train, val, test_unseen, test_seen, keys=('class_attr',),
                                                       feats_range=(0, 1))
feats_dim = len(train['feats'][0])
nb_classes, nb_attributes = train[ATTRS_KEY].shape


#%%
import torch.nn.functional as NN
net = Net().to(device)


opt = torch.optim.Adam(net.parameters(), lr=.00001)

train_dataset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6, pin_memory=False, drop_last=False)

A = torch.tensor(train[ATTRS_KEY]).float().to(device)
A_bin = torch.tensor(train['class_attr_bin']).float().to(device)

def append_loss(lst, loss):
    lst.append(loss.detach().cpu())


def non_diag_indices(rows_cols):
    diag_idx = SortedSet([(i, i) for i in range(rows_cols)])
    all_idx = SortedSet([tuple(l) for l in torch.triu_indices(rows_cols, rows_cols, -rows_cols).transpose(0, 1).numpy().tolist()])
    non_diag_idx = all_idx.difference(diag_idx)
    non_diag_idx = torch.Tensor(non_diag_idx).transpose(0, 1).long()
    return non_diag_idx


# run_test(net, (2048, 1024, 512,), train, test_unseen,
#          nb_gen_class_samples=200, adapt_epochs=4, adapt_lr=.0001, adapt_bs=128, device=device)

non_diag_idx = non_diag_indices(nb_attributes)
random_a_cntx_loss = torch.log(torch.tensor([nb_attributes]).float()).to(device)
random_a_dis_loss = torch.log(torch.tensor([nb_attributes*(nb_attributes-1)]).float()).to(device)

for ep in range(nb_epochs):
    print(f"Running Epoch {ep+1}/{nb_epochs}")
    L_logits, L_cls_z, L_cls_z1, L_z, L_a, L_a_cntx, L_a_dis = [], [], [], [], [], [], []
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        a = A[y]
        a_bin = A_bin[y]

        net.zero_grad()
        logits, logits1, z, z1, a1, a1_dis, a1_cntx = net.forward(x, a_bin)
        #x1, a1, a1_dis, a1_cntx = net.forward(x)



        a_dis = a[:,None,:].repeat(1, nb_attributes, 1)
        K = non_diag_idx[None, :,:].repeat(a.shape[0], 1, 1)
        a_dis = a_dis[:, non_diag_idx[0], non_diag_idx[1]].reshape(a.shape[0], nb_attributes, nb_attributes-1)

        loss_z = NN.mse_loss(z1, z)
        loss_z1_cce = NN.cross_entropy(logits1, y)
        loss_z_cce = NN.cross_entropy(logits, y)
        #loss_logits = NN.mse_loss(logits1, logits)
        loss_logits = NN.mse_loss(logits, logits1)

        if ATTRS_KEY == 'class_attr_bin':
            loss_a = NN.binary_cross_entropy(a1, a)
            loss_a_cntx = torch.min(NN.binary_cross_entropy(a1_cntx, a),
                                    random_a_cntx_loss).mean()
            loss_a_dis = torch.min(NN.binary_cross_entropy(a1_dis.contiguous().view([a1_dis.shape[0], -1]),
                                                           a_dis.contiguous().view([a_dis.shape[0], -1])),
                                   random_a_dis_loss).mean()
        elif ATTRS_KEY == 'class_attr':
            loss_a = NN.mse_loss(a1, a)
            loss_a_cntx = NN.l1_loss(a1_cntx, a)
            loss_a_dis = NN.l1_loss(a1_dis, a_dis)
        else:
            raise ValueError()


        l =  3. *loss_z
        l += 1. *loss_z1_cce
        l += 1. *loss_z_cce
        l += 5. *loss_logits
        l += 1. *loss_a
        l += 1. *loss_a_cntx
        l += 1. *loss_a_dis

        l.backward()
        opt.step()
        append_loss(L_cls_z, loss_z_cce); append_loss(L_cls_z1, loss_z1_cce); append_loss(L_logits, loss_logits)
        append_loss(L_z, loss_z); append_loss(L_a_dis, loss_a_dis); append_loss(L_a, loss_a); append_loss(L_a_cntx, loss_a_cntx)

    print(f"  - Loss z:       {torch.stack(L_z).mean():4.5f}")
    print(f"  - CCE z:        {torch.stack(L_cls_z).mean():4.5f}")
    print(f"  - CCE z':       {torch.stack(L_cls_z1).mean():4.5f}")
    print(f"  - Loss logits:  {torch.stack(L_logits).mean():4.5f}")
    print(f"  - Loss a:       {torch.stack(L_a).mean():4.5f}")
    print(f"  - Loss a_dis:   {torch.stack(L_a_dis).mean():4.5f}")
    print(f"  - Loss a_cntx:  {torch.stack(L_a_cntx).mean():4.5f}")
    print("")
    if ep+1 >= 15:
        run_test(net, train, test_unseen, use_infinite_dataset=True,
                 nb_gen_class_samples=100, adapt_epochs=7, adapt_lr=.0004, adapt_bs=128, device=device)
