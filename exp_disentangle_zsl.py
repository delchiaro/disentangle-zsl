from torch.optim import Adam

from data import get_dataset, preprocess_dataset, download_data
import torch

from disentangle.layers import get_fc_net
from disentangle.net import DisentangleGen
from test import TestCfg
from utils import init
from train import run_wgan_train, TrainCfg
from torch.optim.rmsprop import RMSprop


# %%


def main():
    ################### CONFIGURATIONS
    DOWNLOAD_DATA = False
    PREPROCESS_DATA = False
    DATASET = 'AWA2'  # 'CUB'
    ATTRS_KEY = 'class_attr' # 'class_attr_bin'

    device = init(gpu_index=0, seed=42)

    generalized = False
    use_discriminator = False

    adapt_lr = .0002  # .0002
    lr =.000008 # .00001  #
    #lr =.0002 # .00001  #

    train_cfg = TrainCfg(opt_fn=lambda p: Adam(p, lr=lr),
                         discr_opt_fn=lambda p: Adam(p, lr=lr),
                         nb_epochs=100,
                         bs=128,
                         autoencoder_pretrain_epochs=0,
                         test_epoch=1, test_period=1)

    test_cfg = TestCfg(adapt_epochs=3,
                       adapt_lr=adapt_lr,
                       adapt_bs=128,

                       nb_seen_samples_mean=8, nb_seen_samples_std=2,
                       nb_gen_class_samples=400,

                       infinite_dataset=True,
                       threshold=.5)


     ################### START EXP
    if DOWNLOAD_DATA:
        download_data()
    if PREPROCESS_DATA:
        preprocess_dataset(DATASET)
    train, val, test_unseen, test_seen = get_dataset(DATASET, use_valid=False, gzsl=generalized, mean_sub=True, std_norm=True)
    feats_dim = len(train['feats'][0])
    nb_classes, nb_attributes = train[ATTRS_KEY].shape

    generator = DisentangleGen(nb_classes, nb_attributes,
                               feats_dim=feats_dim,
                               ###
                               encoder_units=(1024, 512),
                               attr_encoder_units=32,  # 32
                               cntx_encoder_units=128,
                               ###
                               decoder_units=(512, 1024, 2048),
                               ###
                               classifier_hiddens=(1024, 512,),
                               cntx_classifier_hiddens=(1024,),
                               attr_regr_hiddens=32).to(device)

    discriminator, _ = get_fc_net(feats_dim, [512], 1, device=device)  \
                           if use_discriminator else None, None

    run_wgan_train(generator, train, test_unseen, test_seen, discriminator, train_cfg, test_cfg,
                   device=device, attr_key=ATTRS_KEY)

if __name__ == '__main__':
    main()
