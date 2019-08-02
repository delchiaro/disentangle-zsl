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

def run_test_old(net: DisentangleZSL, train_dict, unseen_test_dict, seen_test_dict=None,
             attrs_key='class_attr', bs=128, nb_epochs= 10, nb_gen_samples=30, nb_random_means=1, threshold=.5,
             classifier_hiddens=(512,), generalized=None):
    if generalized is None:
       generalized = True if seen_test_dict is not None else False

    train_feats = torch.tensor(train_dict['feats']).float()
    train_labels = torch.tensor(train_dict['labels']).long()
    seen_attrs = torch.tensor(train_dict[attrs_key]).float()
    unseen_test_feats = torch.tensor( unseen_test_dict['feats']).float()
    unseen_test_labels = torch.tensor(unseen_test_dict['labels']).long()
    unseen_attrs = torch.tensor(unseen_test_dict[attrs_key]).float()
    if seen_test_dict is not None:
        seen_test_feats = torch.tensor(seen_test_dict['feats']).float()
        seen_test_labels = torch.tensor(seen_test_dict['labels']).long()

    ######## DATA GENERATION/PREPARATION ###########
    device = net.device

    unseen_gen_feats, unseen_gen_labels = net.generate_feats(train_feats, seen_attrs, unseen_attrs, nb_gen_samples,
                                                             threshold, nb_random_mean=nb_random_means, bs=bs)
    if generalized:
        train_labels_offset = int(sorted(set(unseen_gen_labels))[-1]) + 1
        train_labels = torch.tensor(train_labels).long() + train_labels_offset
        seen_test_labels = torch.tensor(seen_test_labels).long() + train_labels_offset
        seen_test_feats = torch.tensor(seen_test_feats).float()
        A = torch.cat([unseen_attrs, seen_attrs]).float()
        unseen_gen_feats = torch.cat([unseen_gen_feats, train_feats])
        unseen_gen_labels = torch.cat([unseen_gen_labels, train_labels])
    else:
        A = unseen_attrs
    A = A.to(device)
    unseen_gen_feats.float()
    unseen_gen_labels.long()


    ######## TRAINING NEW CLASSIFIER ###########
    adapt_net = deepcopy(net)
    adapt_net.reset_classifiers(len(set(NP(unseen_gen_labels))), classifier_hiddens, classifier_hiddens)

    gen_loader = DataLoader(TensorDataset(unseen_gen_feats, unseen_gen_labels), batch_size=bs, num_workers=2, shuffle=True)
    opt = torch.optim.Adam(adapt_net.classifier.parameters(), lr=.001)
    for ep in range(nb_epochs):
        preds = []
        y_trues = []
        losses = []
        for X, Y in gen_loader:
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            decoded, logits, cntx_logits, _ = adapt_net.forward(X, A[Y])
            loss = NN.cross_entropy(logits, Y)
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu())
            preds.append(logits.argmax(dim=1))
            y_trues.append(Y)
        preds = torch.cat(preds)
        y_trues = torch.cat(y_trues)
        acc = (y_trues == preds).float().mean()
        losses = torch.stack(losses).mean()
        print(f"Classifier adaptation - Epoch {ep+1}/{nb_epochs}:   Loss={losses:1.5f}    Acc={acc:1.4f}")

    ######## TEST ON TEST-SET ###########
    unseen_test_feats = torch.tensor(unseen_test_feats).float()
    unseen_test_labels = torch.tensor(unseen_test_labels).long()
    dloaders = [DataLoader(TensorDataset(unseen_test_feats, unseen_test_labels), batch_size=bs, num_workers=2)]
    if generalized:
        dl_seen = DataLoader(TensorDataset(seen_test_feats, seen_test_labels), batch_size=bs, num_workers=2)
        dloaders.append(dl_seen)

    accs = []
    accs2 = []
    losses = []
    losses2 = []
    for test_loader in dloaders:
        preds = []
        preds2 = []
        y_trues = []
        loss = []
        loss2 = []
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device)
            #decoded, logits, _ = net_copy.forward(X, A[Y]) # BUG HERE: I'm using label information during forward.

            #net_copy.enc_predict(attr_enc, cntx_enc, A)

            logits = adapt_net.predict(X, A)
            _, logits2, cntx_logits2, _ = adapt_net.forward(X)
            # logits1 = net_copy.classifier(X)

            l = NN.cross_entropy(logits, Y)
            loss.append(l.detach().cpu())
            preds.append(logits.argmax(dim=1))

            l2 = NN.cross_entropy(logits2, Y)
            loss2.append(l2.detach().cpu())
            preds2.append(logits2.argmax(dim=1))



            y_trues.append(Y)
        preds = torch.cat(preds)
        preds2 = torch.cat(preds2)
        y_trues = torch.cat(y_trues)
        accs.append((y_trues == preds).float().mean())
        accs2.append((y_trues == preds2).float().mean())
        losses.append(torch.stack(loss).mean())
        losses2.append(torch.stack(loss2).mean())

    unseen_acc, unseen_loss = accs[0], losses[0]
    print(f"\nClassifier adaptation - Final Test - Unseen:   Loss={unseen_loss:1.5f}    Acc={unseen_acc:1.4f}\n")
    print(f"\nClassifier adaptation - Final Test - Unseen:   Loss={losses2[0]:1.5f}     Acc={accs2[0]:1.4f}\n")
    if generalized:
        seen_acc, seen_loss = accs[1], losses[1]
        H_acc = 2 * (seen_acc * unseen_acc) /  (seen_acc + unseen_acc)
        print(f"\n                                   -   Seen:   Loss={seen_loss:1.5f}    Acc={seen_acc:1.4f}\n")
        print(f"\n                                   -      H:                  Acc={H_acc:1.4f}\n")




def run_test_new(net: DisentangleZSL, train_dict, unseen_test_dict, seen_test_dict=None,
             attrs_key='class_attr', bs=128, nb_epochs= 10, nb_gen_samples=30, nb_random_means=1, threshold=.5,
             classifier_hiddens=(512,), generalized=None):
    if generalized is None:
       generalized = True if seen_test_dict is not None else False

    train_feats = torch.tensor(train_dict['feats']).float()
    train_labels = torch.tensor(train_dict['labels']).long()
    seen_attrs = torch.tensor(train_dict[attrs_key]).float()
    unseen_test_feats = torch.tensor( unseen_test_dict['feats']).float()
    unseen_test_labels = torch.tensor(unseen_test_dict['labels']).long()
    unseen_attrs = torch.tensor(unseen_test_dict[attrs_key]).float()
    if seen_test_dict is not None:
        seen_test_feats = torch.tensor(seen_test_dict['feats']).float()
        seen_test_labels = torch.tensor(seen_test_dict['labels']).long()

    ######## DATA GENERATION/PREPARATION ###########
    device = net.device

    gen_attr_encs, gen_cntx_encs, gen_attr, gen_labels = net.generate_encs(train_feats, seen_attrs, unseen_attrs, nb_gen_samples,
                                                                           threshold, nb_random_mean=nb_random_means, bs=bs)
    if generalized:
        train_attr_encodings = []
        train_cntx_encodings = []
        for X in DataLoader(TensorDataset(train_feats), batch_size=bs, num_workers=2, shuffle=False):
            attr_enc, cntx_enc = net.encode(X[0].to(device))
            train_attr_encodings.append(attr_enc.detach().cpu())
            train_cntx_encodings.append(cntx_enc.detach().cpu())
        train_attr_encodings = torch.cat(train_attr_encodings)
        train_cntx_encodings = torch.cat(train_cntx_encodings)
        gen_attr_encs = torch.cat([gen_attr_encs, train_attr_encodings])
        gen_cntx_encs = torch.cat([gen_cntx_encs, train_cntx_encodings])
        gen_labels = torch.cat([gen_labels, train_labels])
        gen_attr = torch.cat([gen_attr, seen_attrs[train_labels]])

    ######## TRAINING NEW CLASSIFIER ###########
    adapt_net = deepcopy(net)
    adapt_net.reset_classifiers(len(set(NP(gen_labels))), classifier_hiddens, classifier_hiddens)
    gen_loader = DataLoader(TensorDataset(gen_attr_encs, gen_cntx_encs, gen_attr, gen_labels), batch_size=bs, num_workers=2, shuffle=True)
    opt = torch.optim.Adam(list(adapt_net.classifier.parameters()) + list(adapt_net.pre_decoder.parameters()), lr=.001)
    for ep in range(nb_epochs):
        preds = []
        y_trues = []
        losses = []
        for attr_enc, cntx_enc, attr, Y in gen_loader:
            attr_enc, cntx_enc, attr, Y = [t.to(device) for t in (attr_enc, cntx_enc, attr, Y)]

            opt.zero_grad()
            logits, _ = adapt_net.classify_decode(attr_enc, cntx_enc, attr)
            loss = NN.cross_entropy(logits, Y)
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu())
            preds.append(logits.argmax(dim=1))
            y_trues.append(Y)
        preds = torch.cat(preds)
        y_trues = torch.cat(y_trues)
        acc = (y_trues == preds).float().mean()
        losses = torch.stack(losses).mean()
        print(f"Classifier adaptation - Epoch {ep + 1}/{nb_epochs}:   Loss={losses:1.5f}    Acc={acc:1.4f}")

    ######## TEST ON TEST-SET ###########


    unseen_test_feats = torch.tensor(unseen_test_feats).float()
    unseen_test_labels = torch.tensor(unseen_test_labels).long()
    dloaders = [DataLoader(TensorDataset(unseen_test_feats, unseen_test_labels), batch_size=bs, num_workers=2)]
    A = unseen_attrs
    if generalized:
        dl_seen = DataLoader(TensorDataset(seen_test_feats, seen_test_labels), batch_size=bs, num_workers=2)
        dloaders.append(dl_seen)
        A = torch.cat([A, seen_attrs])
    A = A.to(device)
    accs = []
    accs2 = []
    losses = []
    losses2 = []

    for test_loader in dloaders:
        preds = []
        preds2 = []
        y_trues = []
        loss = []
        loss2 = []
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device)
            #decoded, logits, _ = net_copy.forward(X, A[Y]) # BUG HERE: I'm using label information during forward.

            #net_copy.enc_predict(attr_enc, cntx_enc, A)

            logits = adapt_net.predict(X, A)
            _, logits2, cntx_logits2, _ = adapt_net.forward(X)
            # logits1 = net_copy.classifier(X)

            l = NN.cross_entropy(logits, Y)
            loss.append(l.detach().cpu())
            preds.append(logits.argmax(dim=1))

            l2 = NN.cross_entropy(logits2, Y)
            loss2.append(l2.detach().cpu())
            preds2.append(logits2.argmax(dim=1))



            y_trues.append(Y)
        preds = torch.cat(preds)
        preds2 = torch.cat(preds2)
        y_trues = torch.cat(y_trues)
        accs.append((y_trues == preds).float().mean())
        accs2.append((y_trues == preds2).float().mean())
        losses.append(torch.stack(loss).mean())
        losses2.append(torch.stack(loss2).mean())

    unseen_acc, unseen_loss = accs[0], losses[0]
    print(f"\nClassifier adaptation - Final Test - Unseen:   Loss={unseen_loss:1.5f}    Acc={unseen_acc:1.4f}\n")
    print(f"\nClassifier adaptation - Final Test - Unseen:   Loss={losses2[0]:1.5f}     Acc={accs2[0]:1.4f}\n")
    if generalized:
        seen_acc, seen_loss = accs[1], losses[1]
        H_acc = 2 * (seen_acc * unseen_acc) /  (seen_acc + unseen_acc)
        print(f"\n                                   -   Seen:   Loss={seen_loss:1.5f}    Acc={seen_acc:1.4f}\n")
        print(f"\n                                   -      H:                  Acc={H_acc:1.4f}\n")

#%%

DOWNLOAD_DATA = False
PREPROCESS_DATA = False
DATASET = 'AWA2'
#ATTRS_KEY = 'class_attr_bin'
ATTRS_KEY = 'class_attr'

def main():

    bs = 128
    nb_epochs = 100
    first_test_epoch, test_period = 1, 1
    nb_class_epochs = 3
    generalized = False

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

    train, val, test_unseen, test_seen = get_dataset(DATASET, use_valid=False, gzsl=generalized)

    nb_train_classes = len(set(train['labels']))
    feats_dim = len(train['feats'][0])

    net = DisentangleZSL(nb_classes=train[ATTRS_KEY].shape[1], feats_dim=feats_dim,
                         pre_encoder_units=(1024, 512),  attr_encoder_units=(32,), cntx_encoder_units=(128,),
                         pre_decoder_units=None, decoder_units=(512,1024,2048),
                         classifier_hiddens=(512,1024), cntx_classifier_hiddens=(512,1024),
                         attr_regr_hiddens=(32,)).to(device)
    opt = torch.optim.Adam(net.parameters())

    from torch.utils.data import DataLoader, TensorDataset

    train_dataset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6, pin_memory=False, drop_last=False)

    # print(f"=======================================================")
    # print(f"=========     RANDOM-WEIGHT TEST      =================")
    # print(f"=======================================================")
    # run_test(net, train['feats'], train[ATTRS_KEY], test_unseen['feats'], test_unseen['labels'], test_unseen[ATTRS_KEY],
    #          nb_epochs=nb_class_epochs, nb_gen_samples=nb_gen_samples, nb_random_means=1,
    #          threshold=np.mean(test_unseen[ATTRS_KEY]),
    #          # threshold=50,
    #          classifier_hiddens=None,
    #          )

    A_train_all = torch.tensor(train[ATTRS_KEY]).float().to(device)

    from progressbar import progressbar
    for ep in range(nb_epochs):
        reconstruction_loss = torch.zeros(len(train_loader))
        attribute_rec_loss = torch.zeros(len(train_loader))
        classification_loss = torch.zeros(len(train_loader))
        context_class_loss = torch.zeros(len(train_loader))
        disentangling_loss = torch.zeros(len(train_loader))
        for bi, (X, Y) in enumerate(progressbar(train_loader)):
            X = X.to(device)
            Y = Y.to(device)

            rec_loss, attr_rec_loss, cls_loss, cntx_cls_loss, dis_loss = net.train_step(X, Y, A_train_all, opt, T=1)
            reconstruction_loss[bi] = rec_loss
            attribute_rec_loss[bi] = attr_rec_loss
            classification_loss[bi] = cls_loss
            context_class_loss[bi] = cntx_cls_loss
            disentangling_loss[bi] = dis_loss

        print(f"")
        print(f"=======================================================")
        print(f"Epoch {ep+1}/{nb_epochs} ended: ")
        print(f"=======================================================")
        print(f"Reconstruction Loss: {torch.mean(reconstruction_loss):2.6f}")
        print(f"Attr Reconstr  Loss: {torch.mean(attribute_rec_loss):2.6f}")
        print(f"Classification Loss: {torch.mean(classification_loss):2.6f}")
        print(f"Context Class  Loss: {torch.mean(context_class_loss):2.6f}")
        print(f"Disentangling  Loss: {torch.mean(disentangling_loss):2.6f}")
        print(f"=======================================================")
        print(f"\n\n\n\n")

        X = X.detach().cpu().numpy()
        #rec = net.reconstruct(X, train['class_attr'][Y.detach().cpu().numpy()])

        if (ep+1) == first_test_epoch or ((ep+1) >= first_test_epoch and (ep+1)%test_period==0):
            print(f"=======================================================")
            print(f"=========         STARTING  TEST        ===============")
            print(f"=======================================================")

            run_test_new(net, train, test_unseen, test_seen,
                         nb_epochs=nb_class_epochs, nb_gen_samples=nb_gen_samples, nb_random_means=1,
                         threshold=np.mean(test_unseen[ATTRS_KEY]),
                         #threshold=50,
                         classifier_hiddens=None)


if __name__ == '__main__':
    main()
