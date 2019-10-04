import os
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from data import get_dataset, preprocess_dataset, download_data, normalize_dataset
from disentangle.layers import get_fc_net, GradientReversal, get_1by1_conv1d_net
from utils import init
import torchvision.utils as vutils

#%%
import torch.nn.functional as NN


class Gen(nn.Module):
    def __init__(self, feats_dim, nb_attributes, noise_dim, enc_hiddens=(256, ), z_dim=512, dec_hiddens=(1024, ),
                 lrelu_slope=None):
        super().__init__()
        self.feats_dim = feats_dim
        self.nb_attributes = nb_attributes
        self.noise_dim = noise_dim
        self.z_dim = z_dim

        hidden_activation = nn.LeakyReLU(lrelu_slope) if lrelu_slope is not None else nn.ReLU()
        self.Ea = get_fc_net(nb_attributes + noise_dim, enc_hiddens, z_dim,hidden_activation, hidden_activation)
        # self.Da = get_fc_net(z_dim, (256,), nb_attributes, nn.LeakyReLU(.2), nn.ReLU())
        self.Dx = get_fc_net(z_dim, dec_hiddens, feats_dim, hidden_activation)
        # self.Da_gr = nn.Sequential(GradientReversal(1), self.Da)  # TODO: we need a new Decoder or we share the weights??

    def encode_a(self, a, n):
        z = self.Ea(torch.cat([a, n], dim=1))
        # za_attr, za_cntx = za[:, :self.z_dim], za[:, self.z_dim:]
        # return za_attr, za_cntx
        return z

    def decode_x(self, z):
        #return self.Dx(torch.cat([z_attr, z_cntx], dim=1))
        return self.Dx(z)

    def forward(self, a, n):
        #z_attr, z_cntx = self.encode_a(a, n)
        #gen_x = self.decode_x(z_attr, z_cntx)
        z = self.encode_a(a, n)
        return self.decode_x(z)



################### CONFIGURATIONS
DOWNLOAD_DATA = False
PREPROCESS_DATA = False
DATASET = 'AWA2'  # 'CUB'
ATTRS_KEY = 'class_attr' # 'class_attr_bin'
exp_name = 'fclswgan_disentangle'
device = init(gpu_index=0, seed=42)
bs=256
lr_g = .0001
lr_d = .0001
nb_epochs = 200
d_iters = 6
clamp_lower, clamp_upper = -.01, .01
 ################### START EXP
if DOWNLOAD_DATA:
    download_data()
if PREPROCESS_DATA:
    preprocess_dataset(DATASET)
train, val, test_unseen, test_seen = get_dataset(DATASET, use_valid=False, gzsl=False, mean_sub=False, std_norm=False, l2_norm=False)
train, val, test_unseen, test_seen = normalize_dataset(train, val, test_unseen, test_seen, keys=('class_attr',), feats_range=(0, 1))
feats_dim = len(train['feats'][0])
nb_classes, nb_attributes = train[ATTRS_KEY].shape

train_dataset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6, pin_memory=False, drop_last=True)

A = torch.tensor(train[ATTRS_KEY]).float().to(device)
A_bin = torch.tensor(train['class_attr_bin']).float().to(device)


noise_size = 32
feat_size = 2048

#%%



netGen = Gen(feats_dim, nb_attributes, noise_size, enc_hiddens=(256, ),  z_dim=1024, dec_hiddens=(1024,)).to(device)
netDisX =  get_fc_net(feats_dim, (1024, 512, 256), 1, hidden_activations=nn.ReLU(), device=device)
#netDisA =  get_fc_net(nb_attributes, (int(nb_attributes/2), int(nb_attributes/4)), 1, hidden_activations=nn.LeakyReLU(.02), device=device)


from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam

optGen = RMSprop(netGen.parameters(), lr=lr_g)
optDisX = RMSprop(netDisX.parameters(), lr=lr_d)
#optGen = Adam(netGen.parameters(), lr=lr_g)
#optDisX = Adam(netDisX.parameters(), lr=lr_d)

#optDisA = RMSprop(netDisA.parameters(), lr=lr_d)

#%%
def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad
#%%


input = torch.FloatTensor(bs, feat_size).to(device)
one = torch.FloatTensor([1]).to(device)
mone = one * -1
noise = torch.FloatTensor(bs, noise_size).to(device)
fixed_noise = torch.FloatTensor(bs, noise_size).normal_(0, 1).to(device)


def aplst(lst, data):
    lst.append(data)

gen_iterations = 0
k = 0
skip_d = False
for epoch in range(nb_epochs):
    data_iter = iter(train_loader)
    i = 0
    E_errD_real, E_errD_fake, E_errD, E_errG, E_errR = [], [], [], [], []

    while i < len(train_loader):

        ############################
        # (1) Update D network
        ###########################
        set_requires_grad(netDisX, True)
        #set_requires_grad(netDisA, True)
        set_requires_grad(netGen, False)
        # train the discriminator Diters times
        # if gen_iterations < 25 or gen_iterations % 500 == 0:
        #     Diters = 100
        # else:
        Diters = d_iters
        j = 0
        while j < Diters and i < len(train_loader) and skip_d is False:
            j += 1

            # clamp parameters to a cube
            for p in netDisX.parameters():
                p.data.clamp_(clamp_lower, clamp_upper)

            data = next(data_iter)
            x, y = data[0].to(device), data[1].to(device)
            a = A[y]
            batch_size = x.shape[0]
            i += 1

            # train with real
            netDisX.zero_grad()
            errD_real = netDisX(x).mean()
            errD_real.backward(one)

            # train with fake
            noise.resize_(batch_size, noise_size).normal_(0, 1).to(device)
            gen_x = netGen(a, noise)

            errD_fake = netDisX(gen_x).mean()
            errD_fake.backward(mone)
            optDisX.step()

            errD = errD_real - errD_fake



        if i < len(train_loader):
            skip_d = False
            ############################
            # (2) Update G network
            ###########################
            set_requires_grad(netDisX, False)
            #set_requires_grad(netDisA, False)
            set_requires_grad(netGen, True)

            data = next(data_iter)
            x, y = data[0].to(device), data[1].to(device)
            a = A[y]
            batch_size = x.shape[0]
            i += 1
            gen_iterations += 1

            netGen.zero_grad()
            noise.resize_(batch_size, noise_size).normal_(0, 1).to(device)
            gen_x = netGen(a, noise)
            errG = netDisX(gen_x).mean()
            errR = NN.mse_loss(gen_x, x, reduction='none').mean(dim=0).sum()
            #errR = NN.l1_loss(gen_x, x, reduction='none').mean(dim=0).sum()

            (1*errG + 100*errR).backward()
            optGen.step()

            aplst(E_errD_real, errD_real)
            aplst(E_errD_fake, errD_fake)
            aplst(E_errD, errD)
            aplst(E_errG, errG)
            aplst(E_errR, errR)
            # from tabulate import tabulate
            # print(f'[{epoch}/{nb_epochs}][{i}/{len(train_loader)}][{gen_iterations}]')
            # tab = tabulate([[errD_real, errD_fake, errD, errG, errR]],
            #                headers=['D-real', 'D-fake', 'D', 'G', 'R'])
            # print(tab)
            # print()
        else:
            skip_d = True


        if gen_iterations % 100 == 0:
            path = f'{exp_name}'
            #x = x.mul(0.5).add(0.5)
            os.makedirs(path, exist_ok=True)
            vutils.save_image(x, os.path.join(path, f'{gen_iterations}_x_real.png'))
            x_gen = netGen(a, fixed_noise)
            #fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(x_gen, os.path.join(path, f'{gen_iterations}_x_fake.png'))

    from tabulate import tabulate

    print(f'[{epoch}/{nb_epochs}][{gen_iterations}]')
    E_errD_real = torch.stack(E_errD_real).mean()
    E_errD_fake = torch.stack(E_errD_fake).mean()
    E_errD = torch.stack(E_errD).mean()
    E_errG = torch.stack(E_errG).mean()
    E_errR = torch.stack(E_errR).mean()
    tab = tabulate([[E_errD_real, E_errD_fake, E_errD, E_errG, E_errR]],
                   headers=['D-real', 'D-fake', 'D', 'G', 'R'])
    print(tab)
    print()