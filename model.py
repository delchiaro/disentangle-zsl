import torch
from torch.nn.modules import Module
from torch import nn
from torch.nn import functional as NN
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from layers import get_fc_net, get_1by1_conv1d_net, GradientReversal


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
        return self.nb_classes * self.attr_enc_dim + self.cntx_enc_dim

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device
        return self

    def reset_classifiers(self, nb_classes,  classifier_hiddens=(512, 1024,), cntx_classifier_hiddens=(512, 1024)):
        self.nb_classes = nb_classes
        self.classifier, _ = get_fc_net(self.pre_decoder_out_dim, classifier_hiddens, self.nb_classes, device=self.device)
        self.cntx_classifier = nn.Sequential(GradientReversal(1),
                                                get_fc_net(self.cntx_enc_dim, cntx_classifier_hiddens, self.nb_classes)[0]).to(self.device)

    def __init__(self,
                 nb_classes,
                 feats_dim=2048,
                 pre_encoder_units=(1024, 512),
                 attr_encoder_units=(32,),
                 cntx_encoder_units=(128,),
                 pre_decoder_units=None,
                 decoder_units=(512, 1024, 2048,),
                 classifier_hiddens=(512, 1024,),
                 cntx_classifier_hiddens=(512, 1024),
                 attr_regr_hiddens=(32,)):
        super().__init__()
        self.device = None
        self.feats_dim = feats_dim
        self.nb_classes = nb_classes
        self.attr_enc_dim = attr_encoder_units[-1]
        self.cntx_enc_dim = cntx_encoder_units[-1]

        # ENCODERS #
        self.pre_encoder, pre_enc_out_dim = get_fc_net(self.feats_dim, pre_encoder_units)
        self.attr_encoder, _ = get_fc_net(pre_enc_out_dim, attr_encoder_units[:-1], attr_encoder_units[-1] * self.nb_classes)
        self.cntx_encoder, _ = get_fc_net(pre_enc_out_dim, cntx_encoder_units)

        # ATTRIBUTE DECODER #
        self.attr_decoder, _  = get_1by1_conv1d_net(self.attr_enc_dim, attr_regr_hiddens, 1, out_activation=nn.Sigmoid)

        # FEATURE DECODER/CLASSIFIER #
        self.pre_decoder, self.pre_decoder_out_dim = get_fc_net(self.full_encoding_dim, pre_decoder_units)
        self.decoder, _ = get_fc_net(self.pre_decoder_out_dim, pre_decoder_units, self.feats_dim)
        self.classifier, _ = get_fc_net(self.pre_decoder_out_dim, classifier_hiddens, self.nb_classes)
        self.cntx_classifier = nn.Sequential(GradientReversal(1), get_fc_net(self.cntx_enc_dim, cntx_classifier_hiddens, self.nb_classes)[0])

    def mask_attr_enc(self, attr_enc, attributes):
        bs = attributes.shape[0]
        nb_attr = attributes.shape[1]
        A = attributes.view(bs, nb_attr, 1).expand(bs, nb_attr, self.attr_enc_dim).contiguous().view(bs, nb_attr * self.attr_enc_dim)
        return attr_enc * A

    def encode(self, x):
        pe = self.pre_encoder(x)
        attr_enc = self.attr_encoder(pe)
        cntx_enc = self.cntx_encoder(pe)
        return attr_enc, cntx_enc

    def pre_decode(self, attr_enc, context_enc, attributes=None):
        if attributes is not None:
            attr_enc = self.mask_attr_enc(attr_enc, attributes)
        concatenated = torch.cat([attr_enc, context_enc], dim=-1)
        # concatenated = torch.cat([attr_enc, torch.zeros_like(context_enc)], dim=-1)
        return self.pre_decoder(concatenated)

    def decode(self, attr_enc, context_enc, attributes=None):
        pre_decoded = self.pre_decode(attr_enc, context_enc, attributes)
        return self.decoder(pre_decoded)

    def classify_decode(self, attr_enc, context_enc, attributes=None):
        pre_decoded = self.pre_decode(attr_enc, context_enc, attributes)
        decoded = self.decoder(pre_decoded)
        logits  = self.classifier(pre_decoded)
        return logits, decoded

    def forward(self, x, a=None):
        attr_enc, cntx_enc = self.encode(x)
        logits, decoded = self.classify_decode(attr_enc, cntx_enc, a)
        cntx_logits = self.cntx_classifier(cntx_enc)
        return decoded, logits, cntx_logits, (attr_enc, cntx_enc)


    @staticmethod
    def reconstruction_loss(x, decoded):
        #return ((decoded-x).abs()).sum(dim=1).mean()
        #return NN.mse_loss(decoded, x)
        return NN.l1_loss(decoded, x)

    @staticmethod
    def attr_reconstruction_loss(x, decoded):
        #return ((decoded-x).abs()).sum(dim=1).mean()
        #return NN.mse_loss(decoded, x)
        return NN.l1_loss(decoded, x)


    def disentangle_loss(self, attr_enc, cntx_enc, label, all_attrs, T=1.):
        nb_classes = len(all_attrs)
        bs = len(attr_enc)
        attr_enc_exp = interlaced_repeat(attr_enc, dim=0, times=nb_classes)
        cntx_enc_exp = interlaced_repeat(cntx_enc, dim=0, times=nb_classes)
        #label_exp = interlaced_repeat(label, dim=0, times=nb_classes)
        all_attrs_exp = all_attrs.repeat([bs, 1])
        #
        logits, decoded = self.classify_decode(attr_enc_exp, cntx_enc_exp, all_attrs_exp)
        t = torch.tensor([[t] for t in list(range(nb_classes)) * bs]).to(self.device)
        logits_diags = torch.gather(logits, 1, t).view(bs, nb_classes)
        return NN.cross_entropy(logits_diags/T, label)

    def reconstruct_attributes(self, attr_enc: torch.Tensor):
        attr_enc = attr_enc.view([attr_enc.shape[0], self.attr_enc_dim, self.nb_classes])
        reconstructed_attr = self.attr_decoder(attr_enc)
        return torch.squeeze(reconstructed_attr, dim=1)

    def train_step(self, X, Y, all_attrs, opt: Optimizer, T=1.,
                   rec_mul=100, attr_rec_mul=10, class_mul=1, cntx_class_mul=0, disent_mul=1):
        opt.zero_grad()
        attr = all_attrs[Y]
        decoded, logits, cntx_logits, (attr_enc, cntx_enc) = self.forward(X, attr)
        attr_reconstr = self.reconstruct_attributes(attr_enc)

        reconstruct_loss = NN.mse_loss(decoded, X) # NN.l1_loss(decoded,X)
        attr_reconstruct_loss = NN.l1_loss(attr_reconstr, attr)
        #attr_reconstruct_loss = torch.Tensor([0.]).to(self.device)
        classifier_loss = NN.cross_entropy(logits/T, Y)
        cntx_classifier_loss = NN.cross_entropy(cntx_logits, Y)

        disentangle_loss = self.disentangle_loss(attr_enc, cntx_enc, Y, all_attrs, T)

        loss = rec_mul * reconstruct_loss + \
               attr_rec_mul * attr_reconstruct_loss + \
               class_mul * classifier_loss + \
               cntx_class_mul * cntx_classifier_loss + \
               disent_mul * disentangle_loss
        loss.backward()
        opt.step()
        return reconstruct_loss, attr_reconstruct_loss, classifier_loss, cntx_classifier_loss, disentangle_loss


    def generate_encs(self, train_feats, train_attrs, test_attrs, nb_gen_samples=30, threshold=.5, nb_random_mean=1, bs=128):
        dl = DataLoader(TensorDataset(train_feats), batch_size=bs, shuffle=False)

        attr_encoding = []
        cntx_encoding = []
        for X in dl:
            X = X[0].to(self.device)
            ae, ce = self.encode(X)
            attr_encoding.append(ae.detach().cpu())
            cntx_encoding.append(ce.detach().cpu())
        attr_encoding = torch.cat(attr_encoding)
        cntx_encoding = torch.cat(cntx_encoding)
        attr_encoding = attr_encoding.view([attr_encoding.shape[0], self.nb_classes, self.attr_enc_dim])

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
                        a_enc = attr_encoding[idx, col]
                        perm = torch.randperm(a_enc.size(0))
                        a_enc = a_enc[perm[:nb_random_mean]].mean(dim=0)

                    else: # if I cant find any example in the dataset with the "same activation status" for attribute a.
                        a_enc = torch.zeros([self.attr_enc_dim])
                    gen_attr_enc.append(a_enc)

                gen_attr_encs.append(torch.cat(gen_attr_enc))
                attrs.append(attr)
                labels.append(lbl)
        attrs = torch.stack(attrs)
        labels = torch.tensor(labels).long()
        gen_attr_encs = torch.stack(gen_attr_encs)
        #gen_cntx_encs = cntx_encoding.mean(dim=0).view(1, -1).repeat([len(gen_attr_encs), 1])
        gen_cntx_encs = cntx_encoding[torch.randperm(cntx_encoding.size(0))[:len(gen_attr_encs)]]
        return gen_attr_encs, gen_cntx_encs, attrs, labels


    def generate_feats(self, train_feats, train_attrs, test_attrs, nb_gen_samples=30, threshold=.5, nb_random_mean=1, bs=128):
        gen_attr_encs, gen_cntx_encs, attrs, labels = self.generate_encs(train_feats, train_attrs, test_attrs,
                                                                         nb_gen_samples, threshold, nb_random_mean, bs)
        gen_feats = self.decode(gen_attr_encs.to(self.device), gen_cntx_encs.to(self.device), attrs.to(self.device)).detach().cpu()
        return gen_feats, labels


    def enc_predict(self, attr_enc, cntx_enc, all_attrs):
        nb_classes = len(all_attrs)
        bs = attr_enc.shape[0]
        attr_enc_exp = interlaced_repeat(attr_enc, dim=0, times=nb_classes)
        all_attrs_exp = all_attrs.repeat([bs, 1])
        cntx_enc_exp = interlaced_repeat(cntx_enc, dim=0, times=nb_classes)

        logits, decoded = self.classify_decode(attr_enc_exp, cntx_enc_exp, all_attrs_exp)
        t = torch.tensor([[t] for t in list(range(nb_classes)) * bs]).to(self.device)
        logits_diags = torch.gather(logits, 1, t).view(bs, nb_classes)
        return logits_diags

    def predict(self, X, all_attrs):
        attr_enc, cntx_enc = self.encode(X)
        return self.enc_predict(attr_enc, cntx_enc, all_attrs)

    def cheaty_predict(self, X, Y, all_attrs):
        attr_enc, cntx_enc = self.encode(X)
        A = all_attrs[Y]
        #attr_enc_exp_masked = self.mask_attr_enc(attr_enc, A)
        logits, decoded = self.classify_decode(attr_enc, cntx_enc, all_attrs)
        softmax = NN.softmax(logits, dim=1)
        return softmax