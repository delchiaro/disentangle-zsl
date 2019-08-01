from typing import Union, List, Tuple

from torch.utils.data import DataLoader, TensorDataset

from data import get_dataset, normalize_dataset, preprocess_dataset, download_data
import torch
from torch.nn.modules import Module
from torch import nn

from torch.nn import functional as NN

from model import DisentangleZSL, get_fc_net
from utils import init, set_seed, to_categorical, unison_shuffled_copies, NP
from copy import deepcopy
import numpy as np



def _init_weights(m: Module):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.xavier_uniform(m.bias)
def init_weights(m: Module):
    m.apply(_init_weights)

def run_test(net: DisentangleZSL, train_feats, train_attrs, test_feats, test_labels, test_attrs, bs=128,
             nb_epochs= 10, nb_gen_samples=30, nb_random_means=1, threshold=.5, classifier_hiddens=[512,]):
    device = net.device
    train_feats = torch.tensor(train_feats).float()
    train_feats = torch.tensor(train_feats).float()
    train_attrs = torch.tensor(train_attrs).float()

    test_feats = torch.tensor(test_feats).float()
    test_labels = torch.tensor(test_labels).long()
    test_attrs = torch.tensor(test_attrs).float()

    Xg, Yg = net.generate_feats(train_feats, train_attrs, test_attrs, nb_gen_samples, threshold, nb_random_mean=nb_random_means, bs=bs)

    net_copy = deepcopy(net)
    net_copy.nb_classes = len(test_attrs)
    net_copy.classifier = get_fc_net(net_copy.feats_dim, classifier_hiddens, net_copy.nb_classes).to(net.device)

    dl = DataLoader(TensorDataset(Xg, Yg), batch_size=bs, num_workers=2)
    test_attrs = test_attrs.to(device)
    opt = torch.optim.Adam(net_copy.classifier.parameters(), lr=.001)
    for ep in range(nb_epochs):
        preds = []
        y_trues = []
        losses = []
        for X, Y in dl:
            X = X.to(device)
            Y = Y.to(device)
            opt.zero_grad()
            decoded, logits, _ = net_copy.forward(X, test_attrs[Y])
            loss = NN.cross_entropy(logits, Y)
            loss.backward()
            losses.append(loss.detach().cpu())
            opt.step()
            pred = logits.argmax(dim=1)
            preds.append(pred)
            y_trues.append(Y)
        preds = torch.cat(preds)
        y_trues = torch.cat(y_trues)
        acc = (y_trues == preds).float().mean()
        losses = torch.stack(losses).mean()
        print(f"Classifier adaptation - Epoch {ep+1}/{nb_epochs}:   Loss={losses:1.5f}    Acc={acc:1.4f}")

    dl = DataLoader(TensorDataset(test_feats, test_labels), batch_size=bs, num_workers=2)
    preds = []
    y_trues = []
    losses = []
    for X, Y in dl:
        X = X.to(device)
        Y = Y.to(device)
        decoded, logits, _ = net_copy.forward(X, test_attrs[Y])
        loss = NN.cross_entropy(logits, Y)
        losses.append(loss.detach().cpu())
        pred = logits.argmax(dim=1)
        preds.append(pred)
        y_trues.append(Y)
    preds = torch.cat(preds)
    y_trues = torch.cat(y_trues)
    acc = (y_trues == preds).float().mean()
    losses = torch.stack(losses).mean()
    print(f"\nClassifier adaptation - Final Test:   Loss={losses:1.5f}    Acc={acc:1.4f}\n")

#%%

DOWNLOAD_DATA = False
PREPROCESS_DATA = False
DATASET = 'AWA2'
# ATTRS_KEY = 'class_attr_orig'
ATTRS_KEY = 'class_attr'

def main():

    bs = 128
    nb_epochs = 100
    first_test_epoch = 1
    test_period = 1
    nb_class_epochs = 80

    if DATASET.startswith('AWA'):
        nb_gen_samples = 200
    elif DATASET == 'CUB':
        nb_gen_samples = 20  # 200
    else:
        nb_gen_samples = 20

    if DOWNLOAD_DATA:
        download_data()
    if PREPROCESS_DATA:
        preprocess_dataset(DATASET)



    device = init(gpu_index=0, seed=42)

    train, val, test = get_dataset(DATASET, use_valid=False, gzsl=False)
    nb_train_classes = len(set(train['labels']))
    nb_test_classes = len(set(test['labels']))
    feats_dim = len(train['feats'][0])

    net = DisentangleZSL(nb_attr=train[ATTRS_KEY].shape[1], nb_classes=nb_train_classes, feats_dim=feats_dim,
                         attr_encode_dim=32, cntx_encode_dim=128,
                         encoder_hiddens=(1024, 512), decoder_hiddens=(512, 1024), classifier_hiddens=None).to(device)
    opt = torch.optim.Adam(net.parameters())

    from torch.utils.data import DataLoader, TensorDataset

    train_dataset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
    val_dataset = TensorDataset(torch.tensor(val['feats']).float(), torch.tensor(val['labels']).long()) if val is not None else None
    test_dataset = TensorDataset(torch.tensor(test['feats']).float(), torch.tensor(test['labels']).long())

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6, pin_memory=False, drop_last=False)
    #train_loader_test = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=6, pin_memory=False, drop_last=False)
    #val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=6, pin_memory=False, drop_last=False) if val is not None else None
    #test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=6, pin_memory=False, drop_last=False)



    print(f"=======================================================")
    print(f"=========     RANDOM-WEIGHT TEST      =================")
    print(f"=======================================================")
    run_test(net, train['feats'], train[ATTRS_KEY], test['feats'], test['labels'], test[ATTRS_KEY],
             nb_epochs=nb_class_epochs, nb_gen_samples=nb_gen_samples, nb_random_means=1,
             threshold=np.mean(test[ATTRS_KEY]),
             # threshold=50,
             classifier_hiddens=None,
             )

    A_train_all = torch.tensor(train[ATTRS_KEY]).float().to(device)

    from progressbar import progressbar
    for ep in range(nb_epochs):
        reconstruction_loss = torch.zeros(len(train_loader))
        classification_loss = torch.zeros(len(train_loader))
        disentangling_loss = torch.zeros(len(train_loader))
        for bi, (X, Y) in enumerate(progressbar(train_loader)):
            X = X.float().to(device)
            Y = Y.long().to(device)

            rec_loss, attr_rec_loss, cls_loss, dis_loss = net.train_step(X, Y, A_train_all, opt, T=1)


            reconstruction_loss[bi] = rec_loss
            classification_loss[bi] = cls_loss
            disentangling_loss[bi] = dis_loss

        print(f"")
        print(f"=======================================================")
        print(f"Epoch {ep+1}/{nb_epochs} ended: ")
        print(f"=======================================================")
        print(f"Reconstruction Loss: {torch.mean(reconstruction_loss):2.6f}")
        print(f"Attr Reconstr  Loss: {torch.mean(attr_rec_loss):2.6f}")
        print(f"Classification Loss: {torch.mean(classification_loss):2.6f}")
        print(f"Disentangling  Loss: {torch.mean(disentangling_loss):2.6f}")
        print(f"=======================================================")
        print(f"\n\n\n\n")

        X = X.detach().cpu().numpy()
        #rec = net.reconstruct(X, train['class_attr'][Y.detach().cpu().numpy()])

        if (ep+1) == first_test_epoch or ((ep+1) >= first_test_epoch and (ep+1)%test_period==0):
            print(f"=======================================================")
            print(f"=========         STARTING  TEST        ===============")
            print(f"=======================================================")

            run_test(net, train['feats'], train[ATTRS_KEY], test['feats'], test['labels'], test[ATTRS_KEY],
                     nb_epochs=nb_class_epochs, nb_gen_samples=nb_gen_samples, nb_random_means=1,
                     threshold=np.mean(test[ATTRS_KEY]),
                     #threshold=50,
                     classifier_hiddens=None,
                     )


if __name__ == '__main__':
    main()
