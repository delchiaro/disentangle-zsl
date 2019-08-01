import torch
from torch.nn.modules import Module
from torch import nn
from torch.nn import functional as NN
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset


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


def get_fc_net(input_size: int, hidden_sizes: list, output_size: int=None, hidden_activations=nn.ReLU, out_activation=None):
    hidden_sizes = [] if hidden_sizes is None else hidden_sizes
    hidden_sizes = list(hidden_sizes) + ([output_size] if output_size is not None else [])
    decoder_hiddens = [nn.Linear(input_size, hidden_sizes[0])]
    prev_size = hidden_sizes[0]
    for size in hidden_sizes[1:]:
        if hidden_activations is not None:
            decoder_hiddens.append(hidden_activations())
        decoder_hiddens.append(nn.Linear(prev_size, size))
        prev_size = size
    if out_activation is not None:
        decoder_hiddens.append(out_activation)
    return nn.Sequential(*decoder_hiddens)

def get_1by1_conv1d_net(in_channels: int, hidden_channels: list, output_channels: int=None, hidden_activations=nn.ReLU, out_activation=None):
    hidden_channels = [] if hidden_channels is None else hidden_channels
    hidden_channels = list(hidden_channels) + ([output_channels] if output_channels is not None else [])
    hiddens = []
    prev_channels = in_channels
    for channels in hidden_channels[:-1]:
        hiddens.append(nn.Conv1d(prev_channels, out_channels=channels, kernel_size=1, stride=1))
        prev_channels = channels
        if hidden_activations is not None:
            hiddens.append(hidden_activations())
    hiddens.append(nn.Conv1d(prev_channels, out_channels=hidden_channels[-1], kernel_size=1, stride=1))
    if out_activation is not None:
        hiddens.append(out_activation())
    return nn.Sequential(*hiddens)

class DisentangleZSL(Module):


    @property
    def full_encoding_dim(self):
        return self.nb_attr * self.attr_enc_dim + self.cntx_enc_dim

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device
        return self

    def __init__(self,
                 nb_attr,
                 nb_classes,
                 feats_dim=2048,
                 attr_encode_dim=32,
                 cntx_encode_dim=128,
                 encoder_hiddens=(1024, 512),
                 decoder_hiddens=(512, 1024),
                 classifier_hiddens=(1024,),
                 attr_regr_hiddens=(32,)):
        super().__init__()
        self.device = None
        self.feats_dim = feats_dim
        self.nb_classes = nb_classes
        self.nb_attr = nb_attr
        self.attr_enc_dim = attr_encode_dim
        self.cntx_enc_dim = cntx_encode_dim

        # ATTRIBUTE ENCODER #
        self.pre_encoder = get_fc_net(self.feats_dim, encoder_hiddens)
        self.attr_encoder = get_fc_net(encoder_hiddens[-1], None, self.attr_enc_dim*self.nb_attr)
        self.cntx_encoder = get_fc_net(encoder_hiddens[-1], None, self.cntx_enc_dim)
        self.decoder = get_fc_net(self.full_encoding_dim, decoder_hiddens, self.feats_dim)
        self.classifier = get_fc_net(self.feats_dim, classifier_hiddens, self.nb_attr)
        self.attr_regressors = get_1by1_conv1d_net(self.attr_enc_dim, attr_regr_hiddens, 1, out_activation=nn.Sigmoid)

    def mask_attr_enc(self, attr_enc, attributes):
        bs = attributes.shape[0]
        nb_attr = attributes.shape[1]
        A = attributes.view(bs, nb_attr, 1).expand(bs, nb_attr, self.attr_enc_dim).contiguous().view(bs, nb_attr * self.attr_enc_dim)
        return attr_enc * A

    def encode(self, x):
        pd = self.pre_encoder(x)
        attr_enc = self.attr_encoder(pd)
        cntx_enc = self.cntx_encoder(pd)
        return attr_enc, cntx_enc

    def decode(self, attr_enc, context_enc, attributes=None):
        if attributes is not None:
            attr_enc = self.mask_attr_enc(attr_enc, attributes)
        concatenated = torch.cat([attr_enc, context_enc], dim=-1)
        #concatenated = torch.cat([attr_enc, torch.zeros_like(context_enc)], dim=-1)
        return self.decoder(concatenated)

    def classify(self, feat):
        return self.classifier(feat)

    def forward(self, x, a=None):
        attr_enc, cntx_enc = self.encode(x)
        decoded = self.decode(attr_enc, cntx_enc, a)
        logits = self.classify(decoded)
        return decoded, logits, (attr_enc, cntx_enc)

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

        decoded = self.decode(attr_enc_exp, cntx_enc_exp, all_attrs_exp)
        logits = self.classify(decoded)
        t = torch.tensor([[t] for t in list(range(nb_classes)) * bs]).to(self.device)
        logits_diags = torch.gather(logits, 1, t).view(bs, nb_classes)
        return NN.cross_entropy(logits_diags/T, label)

    def reconstruct_attributes(self, attr_enc: torch.Tensor):
        attr_enc = attr_enc.view([attr_enc.shape[0], self.attr_enc_dim, self.nb_attr])
        reconstructed_attr = self.attr_regressors(attr_enc)
        return torch.squeeze(reconstructed_attr, dim=1)

    def train_step(self, X, Y, all_attrs, opt: Optimizer, T=1.):
        opt.zero_grad()
        attr = all_attrs[Y]
        decoded, logits, (attr_enc, cntx_enc) = self.forward(X, attr)
        attr_reconstr = self.reconstruct_attributes(attr_enc)

        reconstruct_loss = NN.mse_loss(decoded, X) # NN.l1_loss(decoded,X)
        attr_reconstruct_loss = NN.l1_loss(attr_reconstr, attr)
        #attr_reconstruct_loss = torch.Tensor([0.]).to(self.device)
        classifier_loss = NN.cross_entropy(logits/T, Y)
        disentangle_loss = self.disentangle_loss(attr_enc, cntx_enc, Y, all_attrs, T)

        loss = 10 * reconstruct_loss + \
               10 * attr_reconstruct_loss + \
               1 * classifier_loss + \
               1 * disentangle_loss
        loss.backward()
        opt.step()
        return reconstruct_loss, attr_reconstruct_loss, classifier_loss, disentangle_loss


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
        attr_encoding = attr_encoding.view([attr_encoding.shape[0], self.nb_attr, self.attr_enc_dim])

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

        all_feats = self.decode(attr_enc_exp, cntx_enc_exp, all_attrs_exp)
        logits = self.classify(all_feats)
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
        decoded = self.decode(attr_enc, cntx_enc, all_attrs)
        logits = self.classify(decoded)
        softmax = NN.softmax(logits, dim=1)
        return softmax