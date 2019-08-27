from torch.optim import Adam

from data import get_dataset, preprocess_dataset, download_data
import torch

from disentangle.layers import get_fc_net
from disentangle.net import DisentangleNet
from utils import init
from train import run_train, run_wgan_train
from torch.optim.rmsprop import RMSprop


# %%


def main():
    DOWNLOAD_DATA = False
    PREPROCESS_DATA = False
    DATASET = 'AWA2'  # 'CUB'
    # ATTRS_KEY = 'class_attr_bin'
    ATTRS_KEY = 'class_attr'

    device = init(gpu_index=0, seed=42)

    nb_epochs = 100
    bs = 128
    first_test_epoch, test_period = 3, 1
    generalized = False
    use_infinite_dataset = True

    seen_samples_mean = 8
    seen_samples_std = 2

    nb_gen_class_samples = 400  # 400
    adapt_lr = .0002  # .0002
    lr = .00001  # .000008
    nb_class_epochs = 8  # 4
    use_discriminator = False

    if DOWNLOAD_DATA:
        download_data()
    if PREPROCESS_DATA:
        preprocess_dataset(DATASET)
    train, val, test_unseen, test_seen = get_dataset(DATASET, use_valid=False, gzsl=generalized)
    feats_dim = len(train['feats'][0])
    nb_classes, nb_attributes = train[ATTRS_KEY].shape


    net = DisentangleNet(nb_classes, nb_attributes,
                         feats_dim=feats_dim,
                         ###
                         encoder_units=(1024, 512),
                         attr_encoder_units=8,  # 32
                         cntx_encoder_units=128,
                         ###
                         decoder_units=(512, 1024, 2048),
                         ###
                         classifier_hiddens=(1024, 512,),
                         cntx_classifier_hiddens=(1024,),
                         attr_regr_hiddens=32).to(device)

    opt = Adam(net.parameters(), lr=lr)
    # discr_opt = torch.optim.Adam(discriminator.parameters(), lr=lr) if discriminator is not None else None
    discriminator, _ = get_fc_net(feats_dim, [512], 1, device=device)  if use_discriminator else None, None
    discr_opt = Adam(discriminator.parameters(), lr=lr) if use_discriminator else None

    run_wgan_train(net, opt, train, val, test_unseen, test_seen, discriminator, discr_opt,
                   nb_epochs=nb_epochs, bs=bs, device=device,
                   first_test_epoch=first_test_epoch, test_period=test_period,
                   nb_class_epochs=nb_class_epochs, adapt_lr=adapt_lr,
                   seen_samples_mean=seen_samples_mean, seen_samples_std=seen_samples_std,
                   attr_key=ATTRS_KEY,
                   use_infinite_dataset=use_infinite_dataset, nb_gen_class_samples=nb_gen_class_samples)


if __name__ == '__main__':
    main()
