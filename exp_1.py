from torch.optim import Adam

from data import get_dataset, preprocess_dataset, download_data
import torch

from disentangle.layers import get_fc_net
from disentangle.net import DisentangleGen
from test import TestCfg
from utils import init
from train import run_train, TrainCfg
from torch.optim.rmsprop import RMSprop


# %%


def main():
    device = init(gpu_index=0, seed=42)
    DOWNLOAD_DATA = False
    PREPROCESS_DATA = False
    generalized = False
    use_discriminator = False


    ################### CONFIGURATIONS

    DATASET = 'AWA1'  # 'CUB'
    #DATASET = 'CUB'  # 'CUB'
    ATTRS_KEY = 'class_attr' # 'class_attr_bin'
    adapt_lr = .0001  # .0002
    lr =.001
    #lr=.000002

    train_cfg = TrainCfg(opt_fn=lambda p: Adam(p, lr=lr),
                         discr_opt_fn=lambda p: Adam(p, lr=lr),
                         nb_epochs=100,
                         bs=128,
                         autoencoder_pretrain_epochs=0,
                         classifier_pretrain_epochs=0,#50,
                         test_epoch=1, test_period=1)

    test_cfg = TestCfg(adapt_epochs=10,
                       adapt_lr=adapt_lr,
                       adapt_bs=128,
                       nb_gen_class_samples=400,
                       use_infinite_dataset=True
                       )
                       #nb_seen_samples_mean=8, nb_seen_samples_std=2,
                       #,
                       #threshold=.5


     ################### START EXP
    if DOWNLOAD_DATA:
        download_data()
    if PREPROCESS_DATA:
        preprocess_dataset(DATASET)
    train, val, test_unseen, test_seen = get_dataset(DATASET, use_valid=False, gzsl=generalized, mean_sub=False, std_norm=False, l2_norm=False)
    feats_dim = len(train['feats'][0])
    nb_classes, nb_attributes = train[ATTRS_KEY].shape

    generator = DisentangleGen(nb_classes, nb_attributes,
                               feats_dim=feats_dim,
                               ###
                               encoder_units=(4096,), # (1024, 512),
                               attr_encoder_units=32,  # 32
                               cntx_encoder_units=1024,
                               ###
                               decoder_units=(4096, ),  # (512, 1024, 2048),
                               ###
                               classifier_hiddens=None,
                               cntx_classifier_hiddens=(1024,),
                               attr_regr_hiddens=None,
                               ).to(device)


    run_train(generator, train, test_unseen, test_seen, train_cfg, test_cfg, device=device, attr_key=ATTRS_KEY)
    #
    # discriminator, _ = get_fc_net(feats_dim, [4096, ], 1, device=device)  \
    #                        if use_discriminator else None, None
    # run_wgan_train(generator, train, test_unseen, test_seen, discriminator, train_cfg, test_cfg,
    #                device=device, attr_key=ATTRS_KEY)

if __name__ == '__main__':
    main()
