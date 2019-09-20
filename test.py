from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch.nn.functional as NN

from disentangle.layers import get_fc_net
from disentangle.utils import JoinDataLoader, NP
from disentangle.dataset_generator import InfiniteDataset, FrankensteinDataset
from disentangle.net import DisentangleGen, generate_feats



@dataclass
class TestCfg:
    adapt_epochs: int = 5
    adapt_lr: float = .0001
    adapt_bs: int = 128
    nb_gen_class_samples: int = 200
    use_infinite_dataset: bool = True
    nb_seen_samples_mean: float = 8
    nb_seen_samples_std: float = 3
    threshold: float =.5

def run_test(net: DisentangleGen, train_dict, zsl_unseen_test_dict, gzsl_seen_test_dict=None, attrs_key='class_attr', cfg: TestCfg=TestCfg()):
    generalized = True if gzsl_seen_test_dict is not None else False

    train_feats = torch.tensor(train_dict['feats']).float()
    train_labels = torch.tensor(train_dict['labels']).long()
    seen_attrs = torch.tensor(train_dict[attrs_key]).float()
    unseen_test_feats = torch.tensor(zsl_unseen_test_dict['feats']).float()
    unseen_test_labels = torch.tensor(zsl_unseen_test_dict['labels']).long()
    unseen_attrs = torch.tensor(zsl_unseen_test_dict[attrs_key]).float()
    if gzsl_seen_test_dict is not None:
        seen_test_feats = torch.tensor(gzsl_seen_test_dict['feats']).float()
        seen_test_labels = torch.tensor(gzsl_seen_test_dict['labels']).long()

    def test_on_test(new_class_offset):
        ######## TEST ON TEST-SET ###########
        dloaders = [DataLoader(TensorDataset(unseen_test_feats, unseen_test_labels+new_class_offset), batch_size=cfg.adapt_bs, num_workers=2)]
        if generalized:
            dl_seen = DataLoader(TensorDataset(seen_test_feats, seen_test_labels), batch_size=cfg.adapt_bs, num_workers=2)
            dloaders.append(dl_seen)

        classifier_only_acc = []
        classifier_only_loss = []
        for test_loader in dloaders:
            classifier_only_preds = []
            y_trues = []
            classifier_only_losses = []
            for X, Y in test_loader:
                X = X.to(device)
                Y = Y.to(device)

                classifier_only_logits = adapt_net.classifier(X)
                classifier_only_l = NN.cross_entropy(classifier_only_logits, Y)
                classifier_only_losses.append(classifier_only_l.detach().cpu())
                classifier_only_preds.append(classifier_only_logits.argmax(dim=1))

                y_trues.append(Y)
            classifier_only_preds = torch.cat(classifier_only_preds)
            y_trues = torch.cat(y_trues)
            #full_net_preds = torch.cat(full_net_preds)\
            #full_net_acc.append((y_trues == full_net_preds).float().mean())

            from sklearn import metrics
            mean_class_acc = np.mean([metrics.recall_score(NP(y_trues), NP(classifier_only_preds), labels=[k], average=None) for k in sorted(set(NP(y_trues)))])
            #mean_examples_acc = (y_trues == classifier_only_preds).float().mean()
            #mean_class_acc_1 = metrics.balanced_accuracy_score(NP(y_trues), NP(classifier_only_preds))

           # full_net_loss.append(torch.stack(full_net_losses).mean())
            classifier_only_acc.append(mean_class_acc)
            classifier_only_loss.append(torch.stack(classifier_only_losses).mean())

        # full_net_unseen_acc, full_net_unseen_loss = full_net_acc[0], full_net_losses[0]
        #print(f"\nClassifier adaptation - Final Test - Unseen (full-net):        Loss={full_net_unseen_loss:1.5f}     Acc={full_net_unseen_acc:1.4f}\n")

        unseen_acc, unseen_loss = classifier_only_acc[0], classifier_only_loss[0]
        print(f"\nClassifier adaptation - Final Test - Unseen (classifier-only): Loss={unseen_loss:1.5f}    Acc={unseen_acc:1.4f}\n")

        if generalized:
            #seen_acc, seen_loss = classifier_only_acc[0], classifier_only_loss[0]
            seen_acc, seen_loss = classifier_only_acc[1], classifier_only_loss[1]
            H_acc = 2 * (seen_acc * unseen_acc) /  (seen_acc + unseen_acc)
            print(f"\n                                   -   Seen:   Loss={seen_loss:1.5f}    Acc={seen_acc:1.4f}\n")
            print(f"\n                                   -      H:                  Acc={H_acc:1.4f}\n")


    ######## DATA GENERATION/PREPARATION ###########
    device = net.device
    adapt_net = deepcopy(net)
    nb_new_classes = len(unseen_attrs)
    if generalized:
        #use_infinite_dataset=False
        A = torch.cat([seen_attrs,unseen_attrs]).float()
        new_class_offset = len(seen_attrs)
        if cfg.use_infinite_dataset:
            gen_dataset = InfiniteDataset(int(cfg.nb_gen_class_samples * nb_new_classes), net,
                                          train_dict['feats'], train_dict['labels'],
                                          train_dict['attr_bin'], zsl_unseen_test_dict['class_attr_bin'],
                                          new_class_offset=new_class_offset)
            seen_dataset = TensorDataset(train_feats.float(), train_labels.long())
            gen_loader = DataLoader(gen_dataset, cfg.adapt_bs, shuffle=True)
            seen_loader = DataLoader(seen_dataset, cfg.adapt_bs, shuffle=True)
            data_loader = JoinDataLoader(gen_loader, seen_loader)
        else:
            unseen_gen_feats, unseen_gen_labels = net.generate_feats(train_feats, seen_attrs, unseen_attrs, cfg.nb_gen_class_samples,
                                                                     cfg.threshold, nb_random_mean=1, bs=cfg.adapt_bs)
            unseen_gen_labels = unseen_gen_labels + new_class_offset
            gen_dataset = TensorDataset(unseen_gen_feats.float(), unseen_gen_labels.long())
            chosen_train_feats = []
            chosen_train_labels = []
            seen_classes = sorted(set(train_labels.numpy()))
            for cls in seen_classes:
                idx = torch.masked_select(torch.tensor(range(len(train_labels))), train_labels == cls)
                perm = np.random.permutation(len(idx))
                labels = train_labels[idx][perm][:cfg.nb_seen_samples_mean]
                feats = train_feats[idx][perm][:cfg.nb_seen_samples_mean]
                chosen_train_feats.append(feats)
                chosen_train_labels.append(labels)
            chosen_train_feats = torch.cat(chosen_train_feats)
            chosen_train_labels = torch.cat(chosen_train_labels)
            seen_dataset = TensorDataset(chosen_train_feats.float(), chosen_train_labels.long())

            dataset = ConcatDataset([gen_dataset, seen_dataset])
            data_loader = DataLoader(dataset, cfg.adapt_bs, shuffle=True)

    else:
        A = unseen_attrs
        new_class_offset = 0
        if cfg.use_infinite_dataset:
            dataset = InfiniteDataset(int(cfg.nb_gen_class_samples * nb_new_classes),
                                      net,
                                      train_dict['feats'], train_dict['labels'],
                                      train_dict['attr_bin'], zsl_unseen_test_dict['class_attr_bin'],
                                      device=device)
        else:
            unseen_gen_feats, unseen_gen_labels = generate_feats(net.encoder, train_feats, seen_attrs, unseen_attrs, cfg.nb_gen_class_samples,
                                                                 threshold=cfg.threshold, nb_random_mean=1, bs=cfg.adapt_bs)
            dataset = TensorDataset(unseen_gen_feats.float(), unseen_gen_labels.long())
        data_loader = DataLoader(dataset, batch_size=cfg.adapt_bs, num_workers=2, shuffle=True)
    A = A.to(device)



    ######## PREPARE NEW CLASSIFIER ###########
    if generalized:
        adapt_net.augment_classifiers(nb_new_classes)
    else:
        adapt_net.reset_classifiers(A.shape[0])
    optim = torch.optim.Adam(adapt_net.classifier.parameters(), lr=cfg.adapt_lr)

    ######## TRAINING NEW CLASSIFIER ###########
    for ep in range(cfg.adapt_epochs):
        preds = []
        y_trues = []
        losses = []
        for data in data_loader:
            if cfg.use_infinite_dataset:
                if generalized:
                    (attr_enc, cntx_enc, gen_Y), (X, Y) = data
                    attr_enc, cntx_enc, gen_Y, X, Y = (t.to(device) for t in [attr_enc, cntx_enc, gen_Y, X, Y])
                    gen_X = net.decode(attr_enc, cntx_enc)
                    #gen_X = net.decode(attr_enc, cntx_enc, A[gen_Y])
                    k = int(max(np.floor(np.random.normal(cfg.nb_seen_samples_mean, cfg.nb_seen_samples_std)), 0))
                    X = torch.cat([gen_X[:cfg.adapt_bs - k], X[:k]], dim=0)
                    Y = torch.cat([gen_Y[:cfg.adapt_bs - k], Y[:k]], dim=0)

                else:
                    attr_enc, cntx_enc, Y = (t.to(device) for t in data)
                    #X = net.decode(attr_enc, cntx_enc)
                    X = net.decode(attr_enc, cntx_enc, A[Y])
            else:
                X, Y = data
                X, Y = X.to(device), Y.to(device)
            optim.zero_grad()
            #decoded, logits, cntx_logits, _ = adapt_net.forward(X, A[Y])
            logits = adapt_net.classifier(X)
            loss = NN.cross_entropy(logits, Y)
            loss.backward()
            optim.step()
            losses.append(loss.detach().cpu())
            preds.append(logits.argmax(dim=1))
            y_trues.append(Y)
        preds = torch.cat(preds)
        y_trues = torch.cat(y_trues)
        acc = (y_trues == preds).float().mean()
        classifier_losses = torch.stack(losses).mean()
        print(f"Classifier adaptation - Epoch {ep+1}/{cfg.adapt_epochs}:   Loss={classifier_losses:1.5f}    Acc={acc:1.4f}")
        test_on_test(new_class_offset)


from sklearn import metrics




def run_zsl_test(net: DisentangleGen,
                 train_dict,
                 zsl_unseen_test_dict,
                 nb_gen_class_samples=100,
                 adapt_epochs: int = 5,
                 adapt_lr: float = .0001,
                 adapt_bs: int = 128,
                 use_infinite_dataset=True,
                 attrs_key='class_attr',
                 device=None):
    feats_dim = train_dict['feats'].shape[1]
    nb_new_classes = len(zsl_unseen_test_dict['class_attr_bin'])
    A = torch.tensor(zsl_unseen_test_dict[attrs_key]).float().to(device)

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
            # Decode with ALL attributes and take diagonal logit:
            #logit = net.classify_unseen_feats(X, A)

            # Use original img-feats:
            logit = classifier(X)

            loss = NN.cross_entropy(logit, Y)
            losses.append(loss.detach().cpu()); preds.append(logit.argmax(dim=1)); y_trues.append(Y)

        preds = torch.cat(preds); y_trues = torch.cat(y_trues); unseen_loss = torch.stack(losses).mean()
        unseen_acc = np.mean([metrics.recall_score(NP(y_trues), NP(preds), labels=[k], average=None) for k in sorted(set(NP(y_trues)))])
        return unseen_loss, unseen_acc

    ######## DATA GENERATION/PREPARATION ###########
    if use_infinite_dataset:
        dataset = InfiniteDataset(nb_gen_class_samples * nb_new_classes, net.encoder,
                                  train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                  zsl_unseen_test_dict['class_attr_bin'], device=device)
    else:
        dataset = FrankensteinDataset(nb_gen_class_samples, net.encoder,
                                      train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                      zsl_unseen_test_dict['class_attr_bin'], device=device)
    data_loader = DataLoader(dataset, batch_size=adapt_bs, num_workers=2, shuffle=True)

    ######## PREPARE NEW CLASSIFIER ###########
    net = deepcopy(net)
    net.reset_classifiers(nb_new_classes)
    classifier = net.classifier
    #classifier = get_fc_net(feats_dim, classifier_hiddens, output_size=nb_new_classes).to(device)
    optim = torch.optim.Adam(classifier.parameters(), lr=adapt_lr)

    ######## TRAINING NEW CLASSIFIER ###########
    for ep in range(adapt_epochs):
        preds = []
        y_trues = []
        losses = []
        for data in data_loader:
            attr_enc, cntx_enc, Y = data[0].to(device), data[1].to(device), data[2].to(device)
            optim.zero_grad()
            X = net.decode(attr_enc, cntx_enc, A[Y])
            logit = classifier(X)
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
