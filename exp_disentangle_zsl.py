from typing import Union, List, Tuple

from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from data import get_dataset, normalize_dataset, preprocess_dataset, download_data
import torch
from torch.nn.modules import Module
from torch import nn

from torch.nn import functional as NN

from model import DisentangleZSL, get_fc_net
from utils import init, set_seed, to_categorical, unison_shuffled_copies, NP, to
from copy import deepcopy
import numpy as np
from infinitedataset import InfiniteDataset


def _init_weights(m: Module):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.xavier_uniform(m.bias)
def init_weights(m: Module):
    m.apply(_init_weights)


class JoinDataLoader:
    def __init__(self, master_data_loader, slave_data_loader):
        self.master = master_data_loader
        self.slave = slave_data_loader
        self.master_iter = iter(self.master)
        self.slave_iter = iter(self.slave)
        self._init_iters()

    def _init_iters(self):
        self.master_iter = iter(self.master)
        self.slave_iter = iter(self.slave)

    def __iter__(self):
        return self

    def _next_slave(self):
        try:
            return next(self.slave_iter)
        except StopIteration:
            self.slave_iter = iter(self.slave)
            return self._next_slave()

    def _next_master(self):
        try:
            return next(self.master_iter)
        except StopIteration:
            self.master_iter = iter(self.master)
            raise StopIteration

    def __next__(self):  # Python 2: def next(self)
        master_stuff = self._next_master()
        slave_stuff = self._next_slave()
        return master_stuff, slave_stuff


def join_data_loaders(master_data_loader, slave_data_loader):
    master_iter = iter(master_data_loader)
    slave_iter = iter(slave_data_loader)
    while True:
        try:
            gen_batch = next(master_iter)
        except StopIteration:
            break
        try:
            seen_batch = next(slave_iter)
        except StopIteration:
            pass
        yield gen_batch, seen_batch

def run_test(net: DisentangleZSL, train_dict, unseen_test_dict, seen_test_dict=None,
             attrs_key='class_attr', bs=128, nb_epochs=10, perclass_gen_samples=30, threshold=.5,
             use_infinite_dataset=True, generalized=None, seen_samples_mean=4, seen_samples_std=2, lr=.001):
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
    adapt_net = deepcopy(net)
    nb_new_classes = len(unseen_attrs)
    if generalized:
        #use_infinite_dataset=False
        A = torch.cat([seen_attrs,unseen_attrs]).float()
        new_class_offset = len(seen_attrs)
        gen_dataset = InfiniteDataset(int(perclass_gen_samples * nb_new_classes), net,
                                      train_dict['feats'], train_dict['labels'], train_dict['attr'], train_dict['attr_bin'],
                                      unseen_test_dict['class_attr_bin'], new_class_offset=new_class_offset)
        seen_dataset = TensorDataset(train_feats.float(), train_labels.long())
        gen_loader = DataLoader(gen_dataset, bs, shuffle=True)
        seen_loader = DataLoader(seen_dataset, bs, shuffle=True)
        data_loader = JoinDataLoader(gen_loader, seen_loader)


    else:
        A = unseen_attrs
        new_class_offset = 0
        if use_infinite_dataset:
            dataset = InfiniteDataset(int(perclass_gen_samples * nb_new_classes), net,
                                      train_dict['feats'], train_dict['labels'], train_dict['attr'], train_dict['attr_bin'],
                                      unseen_test_dict['class_attr_bin'])
        else:
            unseen_gen_feats, unseen_gen_labels = net.generate_feats(train_feats, seen_attrs, unseen_attrs, perclass_gen_samples,
                                                                     threshold=threshold, nb_random_mean=1, bs=bs)
            dataset = TensorDataset(unseen_gen_feats.float(), unseen_gen_labels.long())
        data_loader = DataLoader(dataset, batch_size=bs, num_workers=2, shuffle=True)
    A = A.to(device)



    ######## PREPARE NEW CLASSIFIER ###########
    if generalized:
        adapt_net.augment_classifiers(nb_new_classes)
    else:
        adapt_net.reset_classifiers(A.shape[0])
    opt = torch.optim.Adam(adapt_net.classifier.parameters(), lr=lr)

    ######## TRAINING NEW CLASSIFIER ###########
    for ep in range(nb_epochs):
        preds = []
        y_trues = []
        losses = []
        for data in data_loader:
            if generalized:
                (attr_enc, cntx_enc, gen_Y), (X, Y) = data
                attr_enc, cntx_enc, gen_Y, X, Y = (t.to(device) for t in [attr_enc, cntx_enc, gen_Y, X, Y])
                gen_X = net.decode(attr_enc, cntx_enc, A[gen_Y])
                #X = gen_X
                #Y = gen_Y
                k = int(max(np.floor(np.random.normal(seen_samples_mean, seen_samples_std)), 0))
                X = torch.cat([gen_X[:bs-k], X[:k]], dim=0)
                Y = torch.cat([gen_Y[:bs-k], Y[:k]], dim=0)
            else:
                if use_infinite_dataset:
                    attr_enc, cntx_enc, Y = (t.to(device) for t in data)
                    X = net.decode(attr_enc, cntx_enc, A[Y])
                else:
                    X, Y = data
                    X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            #decoded, logits, cntx_logits, _ = adapt_net.forward(X, A[Y])
            logits = adapt_net.classifier(X)
            loss = NN.cross_entropy(logits, Y)
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu())
            preds.append(logits.argmax(dim=1))
            y_trues.append(Y)
        preds = torch.cat(preds)
        y_trues = torch.cat(y_trues)
        acc = (y_trues == preds).float().mean()
        full_net_losses = torch.stack(losses).mean()
        print(f"Classifier adaptation - Epoch {ep+1}/{nb_epochs}:   Loss={full_net_losses:1.5f}    Acc={acc:1.4f}")

    ######## TEST ON TEST-SET ###########
    unseen_test_feats = torch.tensor(unseen_test_feats).float()
    unseen_test_labels = torch.tensor(unseen_test_labels).long() + new_class_offset
    dloaders = [DataLoader(TensorDataset(unseen_test_feats, unseen_test_labels), batch_size=bs, num_workers=2)]
    if generalized:
        dl_seen = DataLoader(TensorDataset(seen_test_feats, seen_test_labels), batch_size=bs, num_workers=2)
        dloaders.append(dl_seen)

    full_net_acc = []
    classifier_only_acc = []
    full_net_loss = []
    classifier_only_loss = []
    for test_loader in dloaders:
        full_net_preds = []
        classifier_only_preds = []
        y_trues = []
        full_net_losses = []
        classifier_only_losses = []
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device)
            #decoded, logits, _ = net_copy.forward(X, A[Y]) # BUG HERE: I'm using label information during forward.
            #net_copy.enc_predict(attr_enc, cntx_enc, A)
            #_, logits2, cntx_logits2, _ = adapt_net.forward(X)
            full_net_logits = adapt_net.predict(X, A)
            classifier_only_logits = adapt_net.classifier(X)

            full_net_l = NN.cross_entropy(full_net_logits, Y)
            full_net_losses.append(full_net_l.detach().cpu())
            full_net_preds.append(full_net_logits.argmax(dim=1))

            classifier_only_l = NN.cross_entropy(classifier_only_logits, Y)
            classifier_only_losses.append(classifier_only_l.detach().cpu())
            classifier_only_preds.append(classifier_only_logits.argmax(dim=1))



            y_trues.append(Y)
        full_net_preds = torch.cat(full_net_preds)
        classifier_only_preds = torch.cat(classifier_only_preds)
        y_trues = torch.cat(y_trues)
        full_net_acc.append((y_trues == full_net_preds).float().mean())

        from sklearn import metrics
        #mean_class_acc_1 = metrics.balanced_accuracy_score(NP(y_trues), NP(classifier_only_preds))
        mean_class_acc = np.mean([metrics.recall_score(NP(y_trues), NP(classifier_only_preds), labels=[k], average=None) for k in sorted(set(NP(y_trues)))])
        #mean_examples_acc = (y_trues == classifier_only_preds).float().mean()

        classifier_only_acc.append(mean_class_acc)
        full_net_loss.append(torch.stack(full_net_losses).mean())
        classifier_only_loss.append(torch.stack(classifier_only_losses).mean())

    full_net_unseen_acc, full_net_unseen_loss = full_net_acc[0], full_net_losses[0]
    unseen_acc, unseen_loss = classifier_only_acc[0], classifier_only_loss[0]

    print(f"\nClassifier adaptation - Final Test - Unseen (classifier-only): Loss={unseen_loss:1.5f}    Acc={unseen_acc:1.4f}\n")
    print(f"\nClassifier adaptation - Final Test - Unseen (full-net):        Loss={full_net_unseen_loss:1.5f}     Acc={full_net_unseen_acc:1.4f}\n")
    if generalized:
        #seen_acc, seen_loss = classifier_only_acc[0], classifier_only_loss[0]
        seen_acc, seen_loss = classifier_only_acc[1], classifier_only_loss[1]
        H_acc = 2 * (seen_acc * unseen_acc) /  (seen_acc + unseen_acc)
        print(f"\n                                   -   Seen:   Loss={seen_loss:1.5f}    Acc={seen_acc:1.4f}\n")
        print(f"\n                                   -      H:                  Acc={H_acc:1.4f}\n")




#%%

DOWNLOAD_DATA = False
PREPROCESS_DATA = False
DATASET = 'AWA2' #'CUB'
#ATTRS_KEY = 'class_attr_bin'
ATTRS_KEY = 'class_attr'

def main():

    nb_epochs = 100
    first_test_epoch, test_period = 3, 1
    generalized = True
    use_infinite_dataset=True

    bs = 128
    seen_samples_mean = 4
    seen_samples_std = 2


    if DATASET.startswith('AWA'):
        nb_gen_class_samples = 800 #400
        adapt_lr = .0002 # .0002
        lr = .000008
        lr = .00001
        nb_class_epochs = 6 # 4


    elif DATASET == 'CUB':

        nb_gen_class_samples = 50  # 200
        adapt_lr = .001
        lr = .0002
        nb_class_epochs = 50


    else:
        nb_seen_class_samples = 10
        nb_gen_class_samples = 50

#def run_exp():
    if DOWNLOAD_DATA:
        download_data()
    if PREPROCESS_DATA:
        preprocess_dataset(DATASET)

    device = init(gpu_index=0, seed=42)

    train, val, test_unseen, test_seen = get_dataset(DATASET, use_valid=False, gzsl=generalized)

    nb_train_classes = len(set(train['labels']))
    feats_dim = len(train['feats'][0])

    nb_classes, nb_attributes = train[ATTRS_KEY].shape
    net = DisentangleZSL(nb_classes, nb_attributes, feats_dim=feats_dim,
                         pre_encoder_units=(1024, 512),
                         attr_encoder_units=(8,), # (32, )
                         cntx_encoder_units=(128,),

                         pre_decoder_units=(512, 1024, 2048),
                         decoder_units=None,

                         classifier_hiddens=(1024, 512,),
                         cntx_classifier_hiddens=(1024,),
                         attr_regr_hiddens=(32,)).to(device)


    discriminator, _ = get_fc_net(net.nb_attributes*net.attr_enc_dim, [512], 1, device=device)
    discriminator = None

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    discr_opt = torch.optim.Adam(discriminator.parameters(), lr=lr) if discriminator is not None else None

    from torch.utils.data import DataLoader, TensorDataset

    train_dataset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6, pin_memory=False, drop_last=False)


    A_train_all = torch.tensor(train[ATTRS_KEY]).float().to(device)

    from torch.nn import functional as F
    from progressbar import progressbar
    for ep in range(nb_epochs):
        reconstruction_loss = torch.zeros(len(train_loader))
        attribute_rec_loss = torch.zeros(len(train_loader))
        classification_loss = torch.zeros(len(train_loader))
        context_class_loss = torch.zeros(len(train_loader))
        disentangling_loss = torch.zeros(len(train_loader))
        g_discriminator_loss = torch.zeros(len(train_loader))
        d_real_loss = torch.zeros(len(train_loader))
        d_fake_loss = torch.zeros(len(train_loader))
        for bi, (X, Y) in enumerate(progressbar(train_loader)):
            X = X.to(device)
            Y = Y.to(device)
            attr = A_train_all[Y]
            zeros = torch.zeros([X.shape[0], 1]).to(net.device)
            ones = torch.ones([X.shape[0], 1]).to(net.device)

            #### TRAINING GENERATOR NET
            opt.zero_grad()
            # Forward
            decoded, logits, cntx_logits, (attr_enc, cntx_enc) = net.forward(X, attr)
            attr_reconstr = net.reconstruct_attributes(attr_enc)

            # Computing Losses
            l_reconstr = NN.l1_loss(decoded, X)#*X.shape[1]  # NN.mse_loss(decoded, X)
            l_attr_reconstr = NN.l1_loss(attr_reconstr, attr)#*attr.shape[1]
            # attr_reconstruct_loss = torch.Tensor([0.]).to(self.device)
            l_class = NN.cross_entropy(logits, Y)
            l_cntx_class = NN.cross_entropy(cntx_logits, Y)
            l_disentangle = net.disentangle_loss(attr_enc, cntx_enc, Y, A_train_all)
            l_discr = 0

            # Computing Discriminator Fooling Loss
            if discriminator is not None:
                frnk_attr_enc = net.generate_frankenstain_attr_enc(attr_enc)
                l_discr = F.binary_cross_entropy_with_logits(discriminator(frnk_attr_enc), zeros)

            #loss = 1*l_reconstr + 1*l_attr_reconstr + 1*l_class + 100*l_cntx_class + 2*l_disentangle + 5*l_discr
            loss = 100*l_reconstr + 1*l_attr_reconstr + 1*l_class + 2*l_cntx_class + 2*l_disentangle + 5*l_discr
            loss.backward()
            opt.step()
            attr_enc = attr_enc.detach()

            #### TRAINING DISCRIMINATOR NET
            if discriminator is not None:
                discr_opt.zero_grad()
                #decoded, logits, cntx_logits, (attr_enc, cntx_enc) = net.forward(X, attr)
                frnk_attr_enc = net.generate_frankenstain_attr_enc(attr_enc.detach())
                zeros = torch.zeros([X.shape[0], 1]).to(net.device)
                ones = torch.ones([X.shape[0], 1]).to(net.device)
                l_discr_real = F.binary_cross_entropy_with_logits(discriminator(attr_enc), zeros)
                l_discr_fake = F.binary_cross_entropy_with_logits(discriminator(frnk_attr_enc), ones)
                dloss = (l_discr_real + l_discr_fake)/2
                dloss.backward()
                discr_opt.step()
                discr_opt.zero_grad()
                d_real_loss[bi] = l_discr_real
                d_fake_loss[bi] = l_discr_fake

            reconstruction_loss[bi] = l_reconstr
            attribute_rec_loss[bi] = l_attr_reconstr
            classification_loss[bi] = l_class
            context_class_loss[bi] = l_cntx_class
            disentangling_loss[bi] = l_disentangle
            g_discriminator_loss[bi] = l_discr



        print(f"")
        print(f"=======================================================")
        print(f"Epoch {ep+1}/{nb_epochs} ended: ")
        print(f"=======================================================")
        print(f"Reconstruction Loss: {torch.mean(reconstruction_loss):2.6f}")
        print(f"Attr Reconstr  Loss: {torch.mean(attribute_rec_loss):2.6f}")
        print(f"Classification Loss: {torch.mean(classification_loss):2.6f}")
        print(f"Context Class  Loss: {torch.mean(context_class_loss):2.6f}")
        print(f"Disentangling  Loss: {torch.mean(disentangling_loss):2.6f}")
        print(f"Discriminator  Loss: {torch.mean(g_discriminator_loss):2.6f}")
        print(f"--------------------------------------------------------")
        print(f"D real Loss:  {torch.mean(d_real_loss):2.6f}")
        print(f"D fake Loss:  {torch.mean(d_fake_loss):2.6f}")
        print(f"=======================================================")
        print(f"\n\n\n\n")

        #rec = net.reconstruct(X, train['class_attr'][Y.detach().cpu().numpy()])

        if (ep+1) == first_test_epoch or ((ep+1) >= first_test_epoch and (ep+1)%test_period==0):
            print(f"=======================================================")
            print(f"=========         STARTING  TEST        ===============")
            print(f"=======================================================")

            run_test(net, train, test_unseen, test_seen,
                     nb_epochs=nb_class_epochs, perclass_gen_samples=nb_gen_class_samples,
                     threshold=float(np.mean(test_unseen[ATTRS_KEY])),
                     use_infinite_dataset=use_infinite_dataset,
                     seen_samples_mean=seen_samples_mean, seen_samples_std=seen_samples_std,
                     lr=adapt_lr)


if __name__ == '__main__':
    main()
















#%%
# !!! Deprecated !!!
# def run_test_new(net: DisentangleZSL, train_dict, unseen_test_dict, seen_test_dict=None,
#              attrs_key='class_attr', bs=128, nb_epochs= 10, nb_gen_samples=30, nb_random_means=1, threshold=.5,
#              classifier_hiddens=(512,), generalized=None, lr=.001):
#     if generalized is None:
#        generalized = True if seen_test_dict is not None else False
#
#     train_feats = torch.tensor(train_dict['feats']).float()
#     train_labels = torch.tensor(train_dict['labels']).long()
#     seen_attrs = torch.tensor(train_dict[attrs_key]).float()
#     unseen_test_feats = torch.tensor( unseen_test_dict['feats']).float()
#     unseen_test_labels = torch.tensor(unseen_test_dict['labels']).long()
#     unseen_attrs = torch.tensor(unseen_test_dict[attrs_key]).float()
#     if seen_test_dict is not None:
#         seen_test_feats = torch.tensor(seen_test_dict['feats']).float()
#         seen_test_labels = torch.tensor(seen_test_dict['labels']).long()
#
#     ######## DATA GENERATION/PREPARATION ###########
#     device = net.device
#
#     gen_attr_encs, gen_cntx_encs, gen_attr, gen_labels = net.generate_encs(train_feats, seen_attrs, unseen_attrs, nb_gen_samples,
#                                                                            threshold, nb_random_mean=nb_random_means, bs=bs)
#     if generalized:
#         train_attr_encodings = []
#         train_cntx_encodings = []
#         for X in DataLoader(TensorDataset(train_feats), batch_size=bs, num_workers=2, shuffle=False):
#             attr_enc, cntx_enc = net.encode(X[0].to(device))
#             train_attr_encodings.append(attr_enc.detach().cpu())
#             train_cntx_encodings.append(cntx_enc.detach().cpu())
#         train_attr_encodings = torch.cat(train_attr_encodings)
#         train_cntx_encodings = torch.cat(train_cntx_encodings)
#         gen_attr_encs = torch.cat([gen_attr_encs, train_attr_encodings])
#         gen_cntx_encs = torch.cat([gen_cntx_encs, train_cntx_encodings])
#         gen_labels = torch.cat([gen_labels, train_labels])
#         gen_attr = torch.cat([gen_attr, seen_attrs[train_labels]])
#
#     ######## TRAINING NEW CLASSIFIER ###########
#     adapt_net = deepcopy(net)
#     adapt_net.reset_classifiers(len(set(NP(gen_labels))), classifier_hiddens, classifier_hiddens)
#     gen_loader = DataLoader(TensorDataset(gen_attr_encs, gen_cntx_encs, gen_attr, gen_labels), batch_size=bs, num_workers=2, shuffle=True)
#     opt = torch.optim.Adam(list(adapt_net.classifier.parameters()) + list(adapt_net.pre_decoder.parameters()), lr=lr)
#     for ep in range(nb_epochs):
#         preds = []
#         y_trues = []
#         losses = []
#         for attr_enc, cntx_enc, attr, Y in gen_loader:
#             attr_enc, cntx_enc, attr, Y = [t.to(device) for t in (attr_enc, cntx_enc, attr, Y)]
#
#             opt.zero_grad()
#             logits, _ = adapt_net.classify_decode(attr_enc, cntx_enc, attr)
#             loss = NN.cross_entropy(logits, Y)
#             loss.backward()
#             opt.step()
#             losses.append(loss.detach().cpu())
#             preds.append(logits.argmax(dim=1))
#             y_trues.append(Y)
#         preds = torch.cat(preds)
#         y_trues = torch.cat(y_trues)
#         acc = (y_trues == preds).float().mean()
#         losses = torch.stack(losses).mean()
#         print(f"Classifier adaptation - Epoch {ep + 1}/{nb_epochs}:   Loss={losses:1.5f}    Acc={acc:1.4f}")
#
#     ######## TEST ON TEST-SET ###########
#
#
#     unseen_test_feats = torch.tensor(unseen_test_feats).float()
#     unseen_test_labels = torch.tensor(unseen_test_labels).long()
#     dloaders = [DataLoader(TensorDataset(unseen_test_feats, unseen_test_labels), batch_size=bs, num_workers=2)]
#     A = unseen_attrs
#     if generalized:
#         dl_seen = DataLoader(TensorDataset(seen_test_feats, seen_test_labels), batch_size=bs, num_workers=2)
#         dloaders.append(dl_seen)
#         A = torch.cat([A, seen_attrs])
#     A = A.to(device)
#     accs = []
#     accs2 = []
#     losses = []
#     losses2 = []
#
#     for test_loader in dloaders:
#         preds = []
#         preds2 = []
#         y_trues = []
#         loss = []
#         loss2 = []
#         for X, Y in test_loader:
#             X = X.to(device)
#             Y = Y.to(device)
#             #decoded, logits, _ = net_copy.forward(X, A[Y]) # BUG HERE: I'm using label information during forward.
#
#             #net_copy.enc_predict(attr_enc, cntx_enc, A)
#
#             logits = adapt_net.predict(X, A)
#             _, logits2, cntx_logits2, _ = adapt_net.forward(X)
#             # logits1 = net_copy.classifier(X)
#
#             l = NN.cross_entropy(logits, Y)
#             loss.append(l.detach().cpu())
#             preds.append(logits.argmax(dim=1))
#
#             l2 = NN.cross_entropy(logits2, Y)
#             loss2.append(l2.detach().cpu())
#             preds2.append(logits2.argmax(dim=1))
#
#
#
#             y_trues.append(Y)
#         preds = torch.cat(preds)
#         preds2 = torch.cat(preds2)
#         y_trues = torch.cat(y_trues)
#         accs.append((y_trues == preds).float().mean())
#         accs2.append((y_trues == preds2).float().mean())
#         losses.append(torch.stack(loss).mean())
#         losses2.append(torch.stack(loss2).mean())
#
#     unseen_acc, unseen_loss = accs[0], losses[0]
#     print(f"\nClassifier adaptation - Final Test - Unseen:   Loss={unseen_loss:1.5f}    Acc={unseen_acc:1.4f}\n")
#     print(f"\nClassifier adaptation - Final Test - Unseen:   Loss={losses2[0]:1.5f}     Acc={accs2[0]:1.4f}\n")
#     if generalized:
#         seen_acc, seen_loss = accs[1], losses[1]
#         H_acc = 2 * (seen_acc * unseen_acc) /  (seen_acc + unseen_acc)
#         print(f"\n                                   -   Seen:   Loss={seen_loss:1.5f}    Acc={seen_acc:1.4f}\n")
#         print(f"\n                                   -      H:                  Acc={H_acc:1.4f}\n")
