import os
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from data import get_dataset, preprocess_dataset, download_data, normalize_dataset
from disentangle.layers import get_fc_net, GradientReversal, get_1by1_conv1d_net
from utils import init
import torchvision.utils as vutils

################### CONFIGURATIONS
DOWNLOAD_DATA = False
PREPROCESS_DATA = False
DATASET = 'AWA2'  # 'CUB'
ATTRS_KEY = 'class_attr' # 'class_attr_bin'
exp_name = 'fclswgan_disentangle'
device = init(gpu_index=1, seed=42)
bs=128
lr_g = .00001
lr_d = .00001
nb_epochs = 100
d_iters = 100
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


#%%
import torch.nn.functional as NN

class Gen(nn.Module):
    def __init__(self, in_features=2048, z_dim=256, noise_dim=128):
        super().__init__()
        self.noise_dim = noise_dim
        self.z_dim = z_dim

        self.Ex = get_fc_net(in_features, (2048, 1024), 2 * z_dim, nn.LeakyReLU(.2), nn.ReLU())
        self.Ea = get_fc_net(nb_attributes + noise_dim, (1024, 512), 2 * z_dim, nn.LeakyReLU(.2), nn.ReLU())
        self.Da = get_fc_net(z_dim, (256,), nb_attributes, nn.LeakyReLU(.2), nn.ReLU())
        self.Dx = get_fc_net(z_dim + z_dim, (1024, 2048), in_features, nn.LeakyReLU(.2), nn.ReLU())
        self.Da_gr = nn.Sequential(GradientReversal(1), self.Da)  # TODO: we need a new Decoder or we share the weights??

    def encode_x(self, x):
        zx = self.Ex(x)
        zx_attr, zx_cntx = zx[:, :self.z_dim], zx[:, self.z_dim:]
        return zx_attr, zx_cntx

    def encode_a(self, a, n):
        za = self.Ea(torch.cat([a,n], dim=1))
        za_attr, za_cntx = za[:, :self.z_dim], za[:, self.z_dim:]
        return za_attr, za_cntx

    def decode_a(self, z_attr):
        return self.Da(z_attr)

    def decode_gr_a(self, z_cntx):
        return self.Da_gr(z_cntx)

    def decode_x(self, z_attr, z_cntx):
        return self.Dx(torch.cat([z_attr, z_cntx], dim=1))

    def forward(self, x, a, n):
        zx_attr, zx_cntx = self.encode_x(x)
        a1_x = self.decode_a(zx_attr)
        a1_gr_x = self.decode_gr_a(zx_cntx)
        x1_x = self.decode_x(zx_attr, zx_cntx)

        za_attr, za_cntx = self.encode_a(a, n)
        a1_a = self.decode_a(za_attr)
        a1_gr_a = self.decode_gr_a(za_cntx)
        x1_a = self.decode_x(za_attr, za_cntx)

        return (x1_x, a1_x, a1_gr_x), (x1_a, a1_a, a1_gr_a)

    def loss(self, x, a, n):
        (x1_x, a1_x, a1_gr_x), (x1_a, a1_a, a1_gr_a) = self.forward(x, a, n)
        loss_x1_x = NN.mse_loss(x1_x, x)
        loss_a1_x = NN.mse_loss(a1_x, a)
        loss_a1_gr_x = NN.mse_loss(a1_gr_x, a)

        loss_x1_a = NN.mse_loss(x1_a, x)
        loss_a1_a = NN.mse_loss(a1_a, a)
        loss_a1_gr_a = NN.mse_loss(a1_gr_a, a)

        return (loss_x1_x, loss_a1_x, loss_a1_gr_x), (loss_x1_a, loss_a1_a, loss_a1_gr_a)

netGen = Gen().to(device)
netDisX =  get_fc_net(feats_dim, (1024, 256), 1, hidden_activations=nn.LeakyReLU(.02), device=device)
netDisA =  get_fc_net(nb_attributes, (int(nb_attributes/2), int(nb_attributes/4)), 1, hidden_activations=nn.LeakyReLU(.02), device=device)


from torch.optim.rmsprop import RMSprop
optGen = RMSprop(netGen.parameters(), lr=lr_g)
optDisA = RMSprop(netDisA.parameters(), lr=lr_d)
optDisX = RMSprop(netDisX.parameters(), lr=lr_d)

#%%
def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad
#%%
noise_size = 128
feat_size = 2048

input = torch.FloatTensor(bs, feat_size).to(device)
one = torch.FloatTensor([1]).to(device)
mone = one * -1
noise = torch.FloatTensor(bs, noise_size).to(device)
fixed_noise = torch.FloatTensor(bs, noise_size).normal_(0, 1).to(device)

gen_iterations = 0
k = 0
skip_d = False
for epoch in range(nb_epochs):
    data_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):

        ############################
        # (1) Update D network
        ###########################
        set_requires_grad(netDisX, True)
        set_requires_grad(netDisA, True)
        set_requires_grad(netGen, False)
        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
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
            errDX_real = netDisX(x).mean()
            errDX_real.backward(one)

            netDisA.zero_grad()
            errDA_real = netDisA(a).mean()
            errDA_real.backward(one)


            # train with fake
            noise.resize_(batch_size, noise_size).normal_(0, 1).to(device)
            (x1_x, a1_x, a1_gr_x), (x1_a, a1_a, a1_gr_a) = netGen(x, a, noise)

            errDX_a = netDisX(x1_a).mean()
            errDX_x = netDisX(x1_x).mean()
            errDX_fake = (errDX_a + errDX_x) / 2
            errDX_fake.backward(mone)
            optDisX.step()

            errDA_a = netDisA(a1_a).mean()
            errDA_x = netDisA(a1_x).mean()
            errDA_fake = (errDA_a + errDA_x) / 2
            errDA_fake.backward(mone)
            optDisA.step()

            errDA = errDA_real - errDA_fake
            errDX = errDX_real - errDX_fake


        if i < len(train_loader):
            skip_d = False
            ############################
            # (2) Update G network
            ###########################
            set_requires_grad(netDisX, False)
            set_requires_grad(netDisA, False)
            set_requires_grad(netGen, True)

            data = next(data_iter)
            x, y = data[0].to(device), data[1].to(device)
            a = A[y]
            batch_size = x.shape[0]
            i += 1
            gen_iterations += 1

            netGen.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(batch_size, noise_size).normal_(0, 1).to(device)
            (x1_x, a1_x, a1_gr_x), (x1_a, a1_a, a1_gr_a) = netGen(x, a, noise)
            errGX_a = netDisX(x1_a).mean()
            errGX_x = netDisX(x1_x).mean()
            errGX = (errGX_a + errGX_x)/2

            errGA_a = netDisA(a1_a).mean()
            errGA_x = netDisA(a1_x).mean()
            errGA = (errGA_a + errGA_x)/2

            errG = (errGX + errGA)/2
            errG.backward()
            optGen.step()

            from tabulate import tabulate
            print(f'[{epoch}/{nb_epochs}][{i}/{len(train_loader)}][{gen_iterations}]')
            tab = tabulate([['   X  ->  (X,A)', errDX_real, errDX_fake, errGX_x, errGA_x],
                            ['(A,n) ->  (X,A)', errDA_real, errDA_fake, errGX_a, errGA_a],],
                           headers=['', 'D-real', 'D-fake', 'Gen-X', 'Gen-A'])
            print(tab)
            print()
        else:
            skip_d = True


        if gen_iterations % 20 == 0:
            path = f'{exp_name}'
            #x = x.mul(0.5).add(0.5)
            os.makedirs(path, exist_ok=True)
            vutils.save_image(x, os.path.join(path, f'{gen_iterations}_x_real.png'))
            vutils.save_image(a, os.path.join(path, f'{gen_iterations}_a_real.png'))
            (x1_x, a1_x, a1_gr_x), (x1_a, a1_a, a1_gr_a) = netGen(x, a, fixed_noise)
            #fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(x1_x, os.path.join(path, f'{gen_iterations}_x_fake_x.png'))
            vutils.save_image(x1_a, os.path.join(path, f'{gen_iterations}_x_fake_a.png'))
            vutils.save_image(a1_x, os.path.join(path, f'{gen_iterations}_a_fake_x.png'))
            vutils.save_image(a1_a, os.path.join(path, f'{gen_iterations}_a_fake_a.png'))

