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
             zsl_unseen_test_dict,
             adapt_bs: int = 128,
             device=None):

        unseen_test_feats = torch.tensor(zsl_unseen_test_dict['feats']).float()
        unseen_test_labels = torch.tensor(zsl_unseen_test_dict['labels']).long()
        dloader = DataLoader(TensorDataset(unseen_test_feats, unseen_test_labels), batch_size=adapt_bs, num_workers=2)
        preds, y_trues = [], []
        all_attr = torch.tensor(zsl_unseen_test_dict['class_attr']).float().to(device)

        predicted_attrs = []
        for X, Y in dloader:
            X = X.to(device)
            (attr_enc, cntx_enc), (a1, a1_dis, a1_cntx) = net.forward(X)
            dists = torch.cdist(a1, all_attr)
            pred = torch.argmin(dists, 1)

            predicted_attrs.append(a1.detach().cpu())
            y_trues.append(Y.detach().cpu())
            preds.append(pred.detach().cpu())

        preds = torch.cat(preds); y_trues = torch.cat(y_trues)
        unseen_acc = np.mean([metrics.recall_score(NP(y_trues), NP(preds), labels=[k], average=None) for k in sorted(set(NP(y_trues)))])

        print(f"1-NN accuracy:  uAcc={unseen_acc:1.5f}")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        in_features = 2048
        pre_enc_units = (2048, 1024, 512)
        attr_enc_units = (256, 64,)
        cntx_enc_units = (256, 128, )
        attr_dec_hiddens = (64, 32)
        cntx_dec_hiddens = (128, 64, 32)

        attr_enc_dim = attr_enc_units[-1] * nb_attributes
        cntx_enc_dim = cntx_enc_units[-1]

        self.enc = DisentangleEncoder(in_features, nb_attributes, pre_enc_units, attr_enc_units, cntx_enc_units)
        self.attr_decoder = get_1by1_conv1d_net(attr_enc_units[-1], attr_dec_hiddens, 1, out_activation=nn.Sigmoid())
        self.dis_attr_decoder = nn.Sequential(GradientReversal(1),
                                              get_1by1_conv1d_net(attr_enc_units[-1], attr_dec_hiddens, nb_attributes-1,
                                                                  out_activation=nn.Sigmoid()))
        self.cntx_decoder = nn.Sequential(GradientReversal(1),
                                          get_fc_net(cntx_enc_dim, cntx_dec_hiddens, nb_attributes, out_activation=nn.Sigmoid()))


    def forward(self, x, a_bin=None):
        bs = x.shape[0]
        attr_enc, cntx_enc = self.enc(x)
        # if a_bin is not None:
        #     mask = interlaced_repeat(a_bin, dim=1, times=self.enc.attr_enc_dim)
        #     attr_enc = attr_enc*mask

        attr_enc_tensor = attr_enc.view(bs, nb_attributes, -1).transpose(1, 2)
        a1 = self.attr_decoder(attr_enc_tensor).view(bs, -1)
        a1_dis = self.dis_attr_decoder(attr_enc_tensor).transpose(1, 2)
        a1_cntx = self.cntx_decoder(cntx_enc)
        return (attr_enc, cntx_enc), (a1, a1_dis, a1_cntx)



#%%

################### CONFIGURATIONS
DOWNLOAD_DATA = False
PREPROCESS_DATA = False
DATASET = 'AWA2'  # 'CUB'
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

classes = sorted(set(train['labels'].tolist()))
cls_counters = []
for cls in classes:
    cls_counters.append(np.sum(train['labels'] == cls))
cls_counters = torch.tensor(cls_counters).float().to(device)
tot_examples = cls_counters.sum()
#%%
import torch.nn.functional as NN
net = Net().to(device)
opt = torch.optim.Adam(net.parameters(), lr=.0004)

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
    L_x, L_a, L_a_cntx, L_a_dis = [], [], [], []
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        a = A[y]
        a_bin = A_bin[y]

        net.zero_grad()
        (attr_enc, cntx_enc), (a1, a1_dis, a1_cntx) = net.forward(x)


        a_dis = a[:,None,:].repeat(1, nb_attributes, 1)
        K = non_diag_idx[None, :,:].repeat(a.shape[0], 1, 1)
        a_dis = a_dis[:, non_diag_idx[0], non_diag_idx[1]].reshape(a.shape[0], nb_attributes, nb_attributes-1)

        l_cls_weight = cls_counters[y]/tot_examples
        #l_cls_weight = 1
        if ATTRS_KEY == 'class_attr_bin':
            loss_a = NN.binary_cross_entropy(a1, a)
            loss_a_cntx = torch.min(NN.binary_cross_entropy(a1_cntx, a),
                                    random_a_cntx_loss).mean()
            loss_a_dis = torch.min(NN.binary_cross_entropy(a1_dis.contiguous().view([a1_dis.shape[0], -1]),
                                                           a_dis.contiguous().view([a_dis.shape[0], -1])),
                                   random_a_dis_loss).mean()
        elif ATTRS_KEY == 'class_attr':
            loss_a = 100*(NN.mse_loss(a1, a, reduction='none').mean(dim=1) * l_cls_weight).mean()
            loss_a_cntx = 100* (NN.mse_loss(a1_cntx, a, reduction='none').mean(dim=1)  * l_cls_weight).mean()
            loss_a_dis = 100* (NN.mse_loss(a1_dis, a_dis, reduction='none').mean(dim=(1,2)) * l_cls_weight).mean()
        else:
            raise ValueError()

        l = .5*loss_a
        l += .25*loss_a_cntx
        l += .25*loss_a_dis

        l.backward()
        opt.step()
        #append_loss(L_x, loss_x);
        append_loss(L_a_dis, loss_a_dis); append_loss(L_a, loss_a); append_loss(L_a_cntx, loss_a_cntx)

    #print(f"  - Loss x:       {torch.stack(L_x).mean():4.5f}")
    print(f"  - Loss a:       {torch.stack(L_a).mean():4.5f}")
    print(f"  - Loss a_dis:   {torch.stack(L_a_dis).mean():4.5f}")
    print(f"  - Loss a_cntx:  {torch.stack(L_a_cntx).mean():4.5f}")
    print("")
    if ep+1 >= 1:
        run_test(net, test_unseen, device=device)
