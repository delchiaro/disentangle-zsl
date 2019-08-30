from torch import nn
import torch

from data import get_dataset, preprocess_dataset, download_data

from disentangle.layers import get_fc_net, GradientReversal, get_1by1_conv1d_net
from disentangle.net import DisentangleEncoder
from utils import init



################### CONFIGURATIONS
DOWNLOAD_DATA = False
PREPROCESS_DATA = False
DATASET = 'AWA2'  # 'CUB'
ATTRS_KEY = 'class_attr' # 'class_attr_bin'
device = init(gpu_index=0, seed=42)

 ################### START EXP
if DOWNLOAD_DATA:
    download_data()
if PREPROCESS_DATA:
    preprocess_dataset(DATASET)
train, val, test_unseen, test_seen = get_dataset(DATASET, use_valid=False, gzsl=False, mean_sub=False, std_norm=False, l2_norm=False)
feats_dim = len(train['feats'][0])
nb_classes, nb_attributes = train[ATTRS_KEY].shape



#%%

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

#%%
in_features = 2048

encoder, _ = get_fc_net(in_features, (2048, 1024, 512+128))
x_decoder, _ = get_fc_net(512+128, (1024, 2048), feats_dim)
a_decoder, _ = get_fc_net(512, (1024, 2048), feats_dim)


discriminator, _ = get_fc_net(feats_dim, [512], 1, device=device)

#%%
