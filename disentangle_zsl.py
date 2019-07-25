from typing import Union, List, Tuple

from data import download_preprocess_all, get_dataset, normalize_dataset
import torch, torchvision
from torch.nn.modules import Module
from torch import nn

from torch.nn import functional as NN
from torch.optim.optimizer import Optimizer
from utils import init, set_seed, to_categorical, unison_shuffled_copies, NP
from copy import deepcopy
import numpy as np

def interlaced_repeat(x, dim, times):
    orig_shape = x.shape
    dims = []
    for i in range(dim+1):
        dims.append(x.shape[i])
    dims.append(1)
    for i in range(dim+1, len(x.shape)):
        dims.append(x.shape[i])
    x = x.view(*dims)
    dims[dim+1]=times
    x = x.expand(dims).contiguous()
    dims = list(orig_shape)
    dims[dim] = dims[dim]*times
    x = x.view(dims)
    return x

class DisentangleZSL(Module):
    @property
    def full_encoding_dim(self):
        return self.nb_attr * self.attr_enc_dim + self.cntx_enc_dim

    def __init__(self, nb_attr, nb_classes, feats_dim=2048,
                 attr_encoder_dims=(1024, 16),
                 cntx_encoder_dims=(1024, 128),
                 decoder_dims=(512,),
                 classifier_dims=(512,)):
        super().__init__()
        self.nb_attr = nb_attr
        self.attr_enc_dim = attr_encoder_dims[-1]
        self.cntx_enc_dim = cntx_encoder_dims[-1]

        # ATTRIBUTE ENCODER #
        attr_encoder_dims = attr_encoder_dims[:-1]
        prev_dim = feats_dim
        attr_encoder_hiddens = []
        if len(attr_encoder_dims) > 0:
            for dim in attr_encoder_dims:
                attr_encoder_hiddens.append(nn.Linear(prev_dim, dim))
                attr_encoder_hiddens.append(nn.ReLU())
                prev_dim = dim
        attr_encoder_hiddens.append(nn.Linear(prev_dim, self.attr_enc_dim * self.nb_attr))
        self.attr_encoder = nn.Sequential(*attr_encoder_hiddens)

        # CONTEXT ENCODER #
        prev_dim = cntx_encoder_dims[0]
        cntx_encoder_hiddens = [nn.Linear(feats_dim, prev_dim)]
        for dim in cntx_encoder_dims[1:]:
            cntx_encoder_hiddens.append(nn.ReLU())
            cntx_encoder_hiddens.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.cntx_encoder = nn.Sequential(*cntx_encoder_hiddens)

        # CLASSIFIER #
        classifier_dims = list(classifier_dims) + [nb_classes]
        prev_dim = classifier_dims[0]
        classifier_hiddens = [nn.Linear(self.full_encoding_dim, prev_dim)]
        for dim in classifier_dims[1:]:
            classifier_hiddens.append(nn.ReLU())
            classifier_hiddens.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.classifier = nn.Sequential(*classifier_hiddens)

        # DECODER #
        decoder_dims = list(decoder_dims) + [feats_dim]
        prev_dim = decoder_dims[0]
        decoder_hiddens = [nn.Linear(self.full_encoding_dim, prev_dim)]
        for dim in decoder_dims[1:]:
            decoder_hiddens.append(nn.ReLU())
            decoder_hiddens.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.decoder = nn.Sequential(*decoder_hiddens)

    def mask_attr_enc(self, attr_enc, attributes):
        bs = attributes.shape[0]
        nb_attr = attributes.shape[1]
        A = attributes.view(bs, nb_attr, 1).expand(bs, nb_attr, self.attr_enc_dim).contiguous().view(bs, nb_attr * self.attr_enc_dim)
        return attr_enc * A

    def encode(self, x):
        attr_enc = self.attr_encoder(x)
        context_enc = self.cntx_encoder(x)
        return attr_enc, context_enc

    def decode(self, attr_enc, context_enc, attributes):
        masked_attr_enc = self.mask_attr_enc(attr_enc, attributes)
        concatenated = torch.cat([masked_attr_enc, context_enc], dim=-1)
        return self.decoder(concatenated)

    def classify(self, attr_enc, context_enc, attributes):
        masked_attr_enc = self.mask_attr_enc(attr_enc, attributes)
        concatenated = torch.cat([masked_attr_enc, context_enc], dim=-1)
        return self.classifier(concatenated)

    def forward(self, x, a):
        self.decode(self.encode(x, a))

    def cross_forward(self, x1, a1, x2, a2):
        e1, c1 = self.encode(x1, a1)
        e2, c2 = self.encode(x2, a2)
        d1 = self.decode(e1, c2)
        d2 = self.decode(e2, c1)
        return d1, d2

    def reconstruction_loss(self, attr_enc, cntx_enc, attr, x):
        decoded = self.decode(attr_enc, cntx_enc, attr)
        return NN.mse_loss(decoded, x)

    def classification_loss(self, attr_enc, cntx_enc, attr, label):
        logits = self.classify(attr_enc, cntx_enc, attr)
        return NN.cross_entropy(logits, label)

    def disentangle_losses(self, attr_enc_1, cntx_enc_1, attr1, label1, attr_enc_2, cntx_enc_2, attr2, label2):
        loss1 = self.classification_loss(attr_enc_1, cntx_enc_2, attr1, label1)
        loss2 = self.classification_loss(attr_enc_2, cntx_enc_1, attr2, label2)
        return loss1, loss2

    def train_step(self, X1, Y1, X2, Y2, all_attrs, opt: Optimizer):
        alpha = 10.
        beta = 1.
        gamma = 1.

        attr1 = all_attrs[Y1]
        attr2 = all_attrs[Y2]
        opt.zero_grad()
        attr_enc_1, cntx_enc_1 = self.encode(X1)
        attr_enc_2, cntx_enc_2 = self.encode(X2)

        reconstruct_loss_1 = self.reconstruction_loss(attr_enc_1, cntx_enc_1, attr1, X1)
        reconstruct_loss_2 = self.reconstruction_loss(attr_enc_2, cntx_enc_2, attr2, X2)
        classifier_loss_1 = self.classification_loss(attr_enc_1, cntx_enc_1, attr1, Y1)
        classifier_loss_2 = self.classification_loss(attr_enc_2, cntx_enc_2, attr2, Y2)
        disentangle_loss_1 = self.classification_loss(attr_enc_1, cntx_enc_2, attr1, Y1)
        disentangle_loss_2 = self.classification_loss(attr_enc_2, cntx_enc_1, attr2, Y2)

        reconstruct_loss = (reconstruct_loss_1 + reconstruct_loss_2) / 2
        classifier_loss = (classifier_loss_1 + classifier_loss_2) / 2
        disentangle_loss = (disentangle_loss_1 + disentangle_loss_2) / 2

        loss = alpha * reconstruct_loss + beta * classifier_loss + gamma * disentangle_loss
        loss.backward()
        opt.step()

        return reconstruct_loss, classifier_loss, disentangle_loss




    def predict(self, X, all_attrs):
        attr_enc, cntx_enc = self.encode(X)
        nb_attrs = len(all_attrs)
        bs = attr_enc.shape[0]
        attr_enc_exp = interlaced_repeat(attr_enc, dim=0, times=nb_attrs)
        all_attrs_exp = all_attrs.repeat([bs, 1])
        cntx_enc_exp = interlaced_repeat(cntx_enc, dim=0, times=nb_attrs)

        attr_enc_exp_masked = self.mask_attr_enc(attr_enc_exp, all_attrs_exp)
        logits = self.classify(attr_enc_exp_masked, cntx_enc_exp, all_attrs_exp)
        softmax = NN.softmax(logits)
        #M = NN.softmax(logits).view(int(logits.shape[0] / 150), logits.shape[1], 150)
        #t = torch.tensor([[i, t] for i, t in zip(range(0, 150*17), list(range(150)) * 17)]).to(device)
        t = torch.tensor([[t] for t in list(range(nb_attrs)) * bs]).to(device)
        softmax = torch.gather(softmax, 1, t).view(bs, nb_attrs)
        return softmax

def prepare_contr_batch(Y, train_dict):
    p = np.random.permutation(range(len(train_dict['feats'])))
    Y_all_shuffeled = deepcopy(train_dict['labels'])[p]
    X_all_shuffeled = deepcopy(train_dict['feats'])[p]

    X_contr = []
    Y_contr = []
    k = 0
    for i in range(len(Y)):
        while 1:
            if k == len(Y_all_shuffeled):
                p = np.random.permutation(range(len(Y_all_shuffeled)))
                Y_all_shuffeled = Y_all_shuffeled[p]
                X_all_shuffeled = X_all_shuffeled[p]
                k = 0

            if Y_all_shuffeled[k] != Y[i]:
                # if 1:
                X_contr.append(X_all_shuffeled[k])
                Y_contr.append(Y_all_shuffeled[k])
                k += 1
                break
            k += 1
    return np.vstack(X_contr), np.array(Y_contr)

#%%
bs = 32

device = init(gpu_index=0, seed=42)
train, val, test = get_dataset('CUB', use_valid=False, gzsl=False)
nb_train_classes = len(set(train['labels']))
nb_test_classes = len(set(test['labels']))
feats_dim = len(train['feats'][0])

net = DisentangleZSL(nb_attr=train['class_attr'].shape[1], nb_classes=nb_train_classes, feats_dim=feats_dim,
                     attr_encoder_dims=(1024, 8),
                     cntx_encoder_dims=(1024, 128),
                     decoder_dims=(2048,),
                     classifier_dims=(2048,)).to(device)
opt = torch.optim.Adam(net.parameters())

from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(torch.tensor(train['feats']).float(),
                              torch.tensor(train['labels']).long())

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6, pin_memory=False, drop_last=False)
#%%
nb_epochs = 10

A = torch.tensor(train['class_attr']).float().to(device)

from progressbar import progressbar
for ep in range(nb_epochs):
    reconstruction_loss = torch.zeros(len(train_loader))
    classification_loss = torch.zeros(len(train_loader))
    disentangling_loss = torch.zeros(len(train_loader))
    for bi, (X, Y) in enumerate(progressbar(train_loader)):
        X2, Y2 = prepare_contr_batch(Y, train)
        X = X.float().to(device)
        Y = Y.long().to(device)
        X2 = torch.tensor(X2).float().to(device)
        Y2 = torch.tensor(Y2).long().to(device)
        #X2 = X
        #Y2 = Y
        rec_loss, cls_loss, dis_loss = net.train_step(X, Y, X2, Y2, A, opt)

        # print(f"Reconstruction Loss: {rec_loss:2.3f}")
        # print(f"Classification Loss: {cls_loss:2.3f}")
        # print(f"Disentangling  Loss: {dis_loss:2.3f}")
        reconstruction_loss[bi] = rec_loss
        classification_loss[bi] = cls_loss
        disentangling_loss[bi] = dis_loss

    print(f"")
    print(f"=======================================================")
    print(f"Epoch {ep+1}/{nb_epochs} ended: ")
    print(f"=======================================================")
    print(f"Reconstruction Loss: {torch.mean(reconstruction_loss):2.3f}")
    print(f"Classification Loss: {torch.mean(classification_loss):2.3f}")
    print(f"Disentangling  Loss: {torch.mean(disentangling_loss):2.3f}")
    print(f"=======================================================")
    print(f"\n\n\n\n")
    sm = net.predict(X, A)
    pred = sm.argmax(dim=1)
    acc = ((pred - Y) == 0).float().mean()
    print(f'Last batch acc: {acc}')

#%%

