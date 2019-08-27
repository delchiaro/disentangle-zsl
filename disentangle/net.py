from copy import deepcopy
import numpy as np

import torch
from torch.nn.modules import Module
from torch import nn
from torch.nn import functional as NN
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from .utils import NP, JoinDataLoader, interlaced_repeat
from .layers import get_fc_net, get_1by1_conv1d_net, GradientReversal



def to_tuple(stuff):
    return (stuff,) if isinstance(stuff, int) else stuff



class DisentangleEncoder(Module):
    def __init__(self,
                 in_features,
                 nb_attributes,
                 pre_encoder_units=(1024, 512),
                 attr_encoder_units=(32,),
                 cntx_encoder_units=(128,)):
        super().__init__()
        self.in_dim = in_features
        self.nb_attr = nb_attributes
        self.attr_enc_dim = attr_encoder_units[-1]
        self.cntx_enc_dim = cntx_encoder_units[-1]
        self.pre_encoder, pre_enc_out_dim = get_fc_net(in_features, pre_encoder_units)
        self.attr_encoder, _ = get_fc_net(pre_enc_out_dim, attr_encoder_units[:-1], attr_encoder_units[-1] * nb_attributes)
        self.cntx_encoder, _ = get_fc_net(pre_enc_out_dim, cntx_encoder_units)

    @property
    def out_dim(self):
        return self.nb_attr * self.attr_enc_dim + self.cntx_enc_dim

    def forward(self, x, a=None):
        return self.encode(x, a)

    def encode(self, x, a=None):
        pe = self.pre_encoder(x)
        attr_enc = self.attr_encoder(pe)
        cntx_enc = self.cntx_encoder(pe)
        if a is not None:
            attr_enc = self.apply_mask(attr_enc, a)
        return attr_enc, cntx_enc

    def full_encode(self, x, a=None):
        attr_enc, cntx_enc = self.encode(x)
        if a is not None:
            attr_enc = self.apply_mask(attr_enc, a)
        return self.join(attr_enc, cntx_enc)

    @staticmethod
    def join(attr_enc, cntx_enc):
        return torch.cat([attr_enc, cntx_enc], dim=-1)

    def divide(self, full_enc):
        attr_enc = full_enc[:self.nb_attr * self.attr_enc_dim]
        cntx_enc = full_enc[self.nb_attr * self.attr_enc_dim:]
        return attr_enc, cntx_enc

    def apply_mask(self, attr_enc, attributes):
        bs = attributes.shape[0]
        A = attributes.view(bs, self.nb_attr, 1).expand(bs, self.nb_attr, self.attr_enc_dim).contiguous().view(bs, self.nb_attr * self.attr_enc_dim)
        return attr_enc * A

    def to_cube(self, attr_enc):
        """ Transform a 2D attr_enc tensor with shape (BS, nb_attr*attr_enc_dim) into a 3D tensor with shape
            (BS, nb_attributes, attr_enc_dim) """
        return attr_enc.view([attr_enc.shape[0], self.nb_attr, self.attr_enc_dim])

    def to_mat(self, cube_attr_enc):
        """ Transform back a 3D cube_attr_enc tensor with shape (BS, nb_attributes, attr_enc_dim) into the original
            2D attr_enc tensor with shape (BS, nb_attributes*attr_enc_dim) """
        return cube_attr_enc.view([cube_attr_enc.shape[0], self.nb_attr,  self.attr_enc_dim])

class DisentangleNet(Module):

    def __init__(self,
                 nb_classes,
                 nb_attributes,
                 feats_dim=2048,
                 encoder_units=(1024, 512),
                 attr_encoder_units=32,
                 cntx_encoder_units=128,
                 decoder_units=(512, 1024, 2048,),
                 classifier_hiddens=(512, 1024,),
                 cntx_classifier_hiddens=(512, 1024),
                 attr_regr_hiddens=32):
        super().__init__()
        attr_encoder_units = to_tuple(attr_encoder_units)
        cntx_encoder_units = to_tuple(cntx_encoder_units)
        encoder_units = to_tuple(encoder_units)
        decoder_units = to_tuple(decoder_units)
        classifier_hiddens = to_tuple(classifier_hiddens)
        cntx_classifier_hiddens = to_tuple(cntx_classifier_hiddens)
        attr_regr_hiddens = to_tuple(attr_regr_hiddens)

        self.device = None
        self.feats_dim = feats_dim
        self.nb_classes = nb_classes
        self.nb_attributes = nb_attributes
        self.attr_enc_dim = attr_encoder_units[-1]
        self.cntx_enc_dim = cntx_encoder_units[-1]

        self.encoder = DisentangleEncoder(feats_dim, nb_attributes, encoder_units, attr_encoder_units, cntx_encoder_units)
        self.decoder, _ = get_fc_net(self.full_encoding_dim, decoder_units, self.feats_dim)
        self.classifier, _ = get_fc_net(self.feats_dim, classifier_hiddens, self.nb_classes)
        self.cntx_classifier = nn.Sequential(GradientReversal(1), get_fc_net(self.cntx_enc_dim, cntx_classifier_hiddens, self.nb_classes)[0])
        self.attr_decoder, _  = get_1by1_conv1d_net(self.attr_enc_dim, attr_regr_hiddens, 1, out_activation=nn.Sigmoid)


    @property
    def full_encoding_dim(self):
        return self.encoder.out_dim

    def forward(self, x, a):
        attr_enc, cntx_enc = self.encode(x)
        masked_attr_enc = self.encoder.apply_mask(attr_enc, a)
        decoded = self.decode(masked_attr_enc, cntx_enc)
        logits  = self.classifier(decoded)
        cntx_logits = self.cntx_classifier(cntx_enc)
        return attr_enc, masked_attr_enc, cntx_enc, decoded, logits, cntx_logits
        #return decoded, logits, cntx_logits, (attr_enc, cntx_enc)

    def encode(self, x, a=None):
        return self.encoder(x, a)

    def decode(self, attr_enc, context_enc, attributes=None):
        if attributes is not None:
            attr_enc = self.encoder.apply_mask(attr_enc, attributes)
        full_enc = self.encoder.join(attr_enc, context_enc)
        return self.decoder(full_enc)

    def disentangle_loss(self, attr_enc, cntx_enc, label, all_attrs, T=1.):
        nb_classes = len(all_attrs)
        bs = len(attr_enc)
        attr_enc_exp = interlaced_repeat(attr_enc, dim=0, times=nb_classes)
        cntx_enc_exp = interlaced_repeat(cntx_enc, dim=0, times=nb_classes)
        #label_exp = interlaced_repeat(label, dim=0, times=nb_classes)
        all_attrs_exp = all_attrs.repeat([bs, 1])
        #
        decoded = self.decode(attr_enc_exp, cntx_enc_exp, all_attrs_exp)
        logits = self.classifier(decoded)
        t = torch.tensor([[t] for t in list(range(nb_classes)) * bs]).to(self.device)
        logits_diags = torch.gather(logits, 1, t).view(bs, nb_classes)
        return NN.cross_entropy(logits_diags/T, label)

    def reconstruct_attributes(self, attr_enc: torch.Tensor):
        cube_attr_enc = self.encoder.to_cube(attr_enc)
        cube_attr_enc = cube_attr_enc.transpose(1, 2)
        reconstructed_attr = self.attr_decoder(cube_attr_enc)
        return torch.squeeze(reconstructed_attr, dim=1)

    def classify_unseen_enc(self, attr_enc, cntx_enc, all_attrs):
        nb_classes = len(all_attrs)
        bs = attr_enc.shape[0]
        attr_enc_exp = interlaced_repeat(attr_enc, dim=0, times=nb_classes)
        all_attrs_exp = all_attrs.repeat([bs, 1])
        cntx_enc_exp = interlaced_repeat(cntx_enc, dim=0, times=nb_classes)
        decoded = self.decode(attr_enc_exp, cntx_enc_exp, all_attrs_exp)
        logits = self.classifier(decoded)
        t = torch.tensor([[t] for t in list(range(nb_classes)) * bs]).to(self.device)
        logits_diags = torch.gather(logits, 1, t).view(bs, nb_classes)
        return logits_diags

    def classify_unseen_feats(self, X, all_attrs):
        attr_enc, cntx_enc = self.encode(X)
        return self.classify_unseen_enc(attr_enc, cntx_enc, all_attrs)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device
        return self

    def reset_classifiers(self, nb_classes):
        self.nb_classes = nb_classes
        self.classifier[-1] = nn.Linear(self.classifier[-1].in_features, nb_classes).to(self.device)
        self.cntx_classifier[-1][-1] = nn.Linear(self.cntx_classifier[-1][-1].in_features, nb_classes).to(self.device)

    def augment_classifiers(self, nb_new_classes):
        self.nb_classes += nb_new_classes
        bias = self.classifier[-1].bias.data.detach().cpu()
        weight = self.classifier[-1].weight.data.detach().cpu()
        augment_bias = torch.zeros([nb_new_classes])
        augment_weight = torch.zeros([nb_new_classes, weight.shape[1]])
        nn.init.xavier_uniform_(augment_weight) #torch.nn.init.xavier_uniform(augment_weight)
        self.classifier[-1].weight = torch.nn.Parameter(torch.cat([weight, augment_weight]).to(self.device))
        self.classifier[-1].bias = torch.nn.Parameter(torch.cat([bias, augment_bias]).to(self.device))
        self.classifier[-1].out_dim = self.nb_classes

        bias = self.cntx_classifier[-1][-1].bias.data.detach().cpu()
        weight = self.cntx_classifier[-1][-1].weight.data.detach().cpu()
        augment_bias = torch.zeros([nb_new_classes])
        augment_weight = torch.zeros([nb_new_classes, weight.shape[1]])
        nn.init.xavier_uniform_(augment_weight)
        self.cntx_classifier[-1][-1].weight = torch.nn.Parameter(torch.cat([weight, augment_weight]).to(self.device))
        self.cntx_classifier[-1][-1].bias = torch.nn.Parameter(torch.cat([bias, augment_bias]).to(self.device))
        self.cntx_classifier[-1][-1].out_dim =  self.nb_classes




def generate_frankenstain_attr_enc(net: DisentangleNet, masked_attr_enc):
    attr_enc_cube = net.encoder.to_cube(masked_attr_enc)
    perms = np.array([np.random.permutation(attr_enc_cube.shape[0]) for _ in range(85)]).transpose([0, 1])
    frnk_attr_enc = torch.stack([attr_enc_cube[perms[k], k, :] for k in range(85)]).transpose(0, 1)
    return frnk_attr_enc.contiguous().view([frnk_attr_enc.shape[0], -1])

def generate_encs(net: DisentangleNet, train_feats, train_attrs, test_attrs, nb_gen_samples=30, threshold=.5, nb_random_mean=1, bs=128):
    dl = DataLoader(TensorDataset(train_feats), batch_size=bs, shuffle=False)
    attr_enc = []
    cntx_enc = []
    for X in dl:
        X = X[0].to(net.device)
        ae, ce = net.encode(X)
        attr_enc.append(ae.detach().cpu())
        cntx_enc.append(ce.detach().cpu())
    attr_enc = torch.cat(attr_enc)
    cntx_enc = torch.cat(cntx_enc)
    attr_enc_cube = net.encoder.to_cube(attr_enc)

    #for attr in test_attrs:
    gen_attr_encs = []
    attrs = []
    labels = []
    for lbl, attr in enumerate(test_attrs):
        for i in range(nb_gen_samples):
            gen_attr_enc = []
            for col, a in enumerate(attr):
                if a >= threshold:
                    idx = (train_attrs[:, col] >= threshold).nonzero().squeeze(dim=1)
                else:
                    idx = (train_attrs[:, col] < threshold).nonzero().squeeze(dim=1)

                if len(idx) > 0:
                    a_enc = attr_enc_cube[idx, col]
                    perm = torch.randperm(a_enc.size(0))
                    a_enc = a_enc[perm[:nb_random_mean]].mean(dim=0)

                else: # if I cant find any example in the dataset with the "same activation status" for attribute a.
                    a_enc = torch.zeros([net.attr_enc_dim])
                gen_attr_enc.append(a_enc)

            gen_attr_encs.append(torch.cat(gen_attr_enc))
            attrs.append(attr)
            labels.append(lbl)
    attrs = torch.stack(attrs)
    labels = torch.tensor(labels).long()
    gen_attr_encs = torch.stack(gen_attr_encs)
    #gen_cntx_encs = cntx_encoding.mean(dim=0).view(1, -1).repeat([len(gen_attr_encs), 1])
    gen_cntx_encs = cntx_enc[torch.randperm(cntx_enc.size(0))[:len(gen_attr_encs)]]
    return gen_attr_encs, gen_cntx_encs, attrs, labels


def generate_feats(net: DisentangleNet, train_feats, train_attrs, test_attrs, nb_gen_samples=30, threshold=.5, nb_random_mean=1, bs=128):
    gen_attr_encs, gen_cntx_encs, attrs, labels = generate_encs(net, train_feats, train_attrs, test_attrs,
                                                                    nb_gen_samples, threshold, nb_random_mean, bs)
    gen_feats = net.decode(gen_attr_encs.to(net.device), gen_cntx_encs.to(net.device), attrs.to(net.device)).detach().cpu()
    return gen_feats, labels

