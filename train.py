from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as NN
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.optimizer import Optimizer

from typing import Callable, Iterable

from disentangle.net import generate_frankenstain_attr_enc, DisentangleGen
from test import run_test, TestCfg



def set_require_grad(net: torch.nn.Module, require_grad: bool):
    for p in net.parameters():
        p.requires_grad = require_grad


@dataclass
class TrainCfg:
    opt_fn: Callable[[Iterable], Optimizer] = lambda p: torch.optim.Adam(p, .0001)
    discr_opt_fn: Callable[[Iterable], Optimizer] = lambda p: torch.optim.Adam(p, .0001)
    nb_epochs: int = 100
    bs: int = 128
    autoencoder_pretrain_epochs: int = 30
    test_epoch: int =5
    test_period: int =5
    wgan_clamp_lower: float =-.01
    wgan_clamp_upper: float =.01


def run_wgan_train(generator: DisentangleGen,
                   train,
                   zsl_test_unseen,
                   gzsl_test_seen=None,
                   discriminator: torch.nn.Module = None,
                   cfg: TrainCfg = TrainCfg(),
                   test_cfg: TestCfg = TestCfg(),
                   attr_key='class_attr',
                   device=None,
                   ):

    train_dataset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
    train_loader = DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True, num_workers=6, pin_memory=False, drop_last=False)
    nb_train_classes = len(set(train['labels']))
    X_mean = np.mean(train['feats'])
    X_std = np.std(train['feats'])
    A_mean = np.mean(train[attr_key])
    A_std = np.std(train[attr_key])

    opt = cfg.opt_fn(generator.parameters())
    if discriminator is not None:
        discr_opt = cfg.discr_opt_fn(discriminator.parameters())

    one = torch.FloatTensor([1]).to(device)
    mone = one * -1
    Diters=25
    A_train_all = torch.tensor(train[attr_key]).float().to(device)
    pretraining = 5
    from time import time
    for ep in range(cfg.nb_epochs):
        time_ep_begin = time()
        reconstruction_loss = []
        attribute_rec_loss = []
        classification_loss = []
        context_class_loss = []
        disentangling_loss = []
        g_loss = []
        d_real_loss = []
        d_fake_loss = []
        d_loss = []

        data_iter = iter(train_loader)
        i=0
        gen_iterations = 0
        while i < len(train_loader):
            if ep < cfg.autoencoder_pretrain_epochs:
                i+=1
                X, Y = next(data_iter)
                X, Y = X.to(device), Y.to(device)
                attr = A_train_all[Y]
                generator.zero_grad()
                masked_attr_enc, cntx_enc = generator.encode(X, attr)
                decoded = generator.decode(masked_attr_enc, cntx_enc)
                attr_decoded = generator.reconstruct_attributes(masked_attr_enc)
                l_reconstr = NN.mse_loss(decoded, X)
                attr_l_reconstruct = NN.mse_loss(attr_decoded, attr)
                l = l_reconstr + attr_l_reconstruct
                l.backward()
                reconstruction_loss.append(l_reconstr.detach().cpu())
                attribute_rec_loss.append(attr_l_reconstruct.detach().cpu())
                opt.step()

            else:
                if discriminator is not None and ep > pretraining:
                    ############################
                    # (1) Update D network
                    ###########################
                    set_require_grad(discriminator, True)
                    set_require_grad(generator, False)
                    # train the discriminator Diters times
                    j = 0
                    while j < Diters and i < len(train_loader):
                        for p in discriminator.parameters():
                            p.data.clamp_(cfg.wgan_clamp_lower, cfg.wgan_clamp_upper)
                        j +=1; i += 1
                        X, Y = next(data_iter)
                        X, Y = X.to(device), Y.to(device)
                        attr =  A_train_all[Y]

                        discriminator.zero_grad()
                        # # train with real
                        # errD_real = torch.mean(discriminator(X))
                        # errD_real.backward(one)

                        masked_attr_enc, cntx_enc = generator.encode(X, attr)
                        frnk_attr_enc = generate_frankenstain_attr_enc(generator, masked_attr_enc)
                        frnk_X = generator.decode(frnk_attr_enc, cntx_enc)
                        decoded_X = generator.decode(masked_attr_enc, cntx_enc)

                        # train with decoded-real
                        errD_real = torch.mean(discriminator(decoded_X))
                        errD_real.backward(one)

                        # train with fake
                        errD_fake = torch.mean(discriminator(frnk_X))
                        errD_fake.backward(mone)
                        errD = errD_real - errD_fake
                        discr_opt.step()

                    d_real_loss.append(errD_real.detach().cpu())
                    d_fake_loss.append(errD_fake.detach().cpu())
                    d_loss.append(errD.detach().cpu())
                    set_require_grad(discriminator, False)
                    set_require_grad(generator, True)

                else:
                    i += 1
                    X, Y = next(data_iter)
                    X, Y = X.to(device), Y.to(device)

                ############################
                # (2) Update G network
                ###########################
                generator.zero_grad()

                # Forward
                attr = A_train_all[Y]
                attr_enc, masked_attr_enc, cntx_enc, decoded, logits, cntx_logits = generator.forward(X, attr)
                attr_reconstr = generator.reconstruct_attributes(masked_attr_enc)


                if discriminator is not None and  ep > pretraining:
                    # Genrate Frankenstain Attributes and Decode fake X
                    frnk_attr_enc = generate_frankenstain_attr_enc(generator, masked_attr_enc)
                    frnk_X = generator.decode(frnk_attr_enc, cntx_enc)

                    # Fake X generator Loss
                    errG = torch.mean(discriminator(frnk_X))
                    errG.backward(one, retain_graph=True)
                    g_loss.append(errG.detach().cpu())


                # Forward Losses
                l_reconstr = NN.mse_loss(decoded, X)  # *X.shape[1]  # NN.mse_loss(decoded, X)
                l_attr_reconstr = NN.mse_loss(attr_reconstr, attr)  # *attr.shape[1]
                l_class = NN.cross_entropy(logits, Y)
                l_cntx_class = NN.cross_entropy(cntx_logits, Y)
                l_disentangle = generator.disentangle_loss(attr_enc, cntx_enc, Y, A_train_all)
                #l = 100*l_reconstr + 1* l_attr_reconstr + 1 * l_class + 2 * l_cntx_class + 2 * l_disentangle
                l = 100*l_reconstr + 1* l_attr_reconstr + 1 * l_class + 0*l_cntx_class + 1 * l_disentangle
                l.backward()

                opt.step()
                gen_iterations += 1

                reconstruction_loss.append(l_reconstr.detach().cpu())
                attribute_rec_loss.append(l_attr_reconstr.detach().cpu())
                classification_loss.append(l_class.detach().cpu())
                context_class_loss.append(l_cntx_class.detach().cpu())
                disentangling_loss.append(l_disentangle.detach().cpu())

        print(f"")
        print(f"=======================================================")
        print(f"Epoch {ep + 1}/{cfg.nb_epochs} ended ({time()-time_ep_begin:3.2f}s): ")
        print(f"=======================================================")
        print(f"Reconstruction Loss: {torch.tensor(reconstruction_loss).mean():2.6f}   X-mean={X_mean:2.6f} X-std={X_std:2.6f}")
        print(f"Attr Reconstr  Loss: {torch.tensor(attribute_rec_loss).mean():2.6f}   A-mean={A_mean:2.6f} A-std={A_std:2.6f}")
        print(f"Classification Loss: {torch.tensor(classification_loss).mean():2.6f}")
        print(f"Context Class  Loss: {torch.tensor(context_class_loss).mean():2.6f}")
        print(f"Disentangling  Loss: {torch.tensor(disentangling_loss).mean():2.6f}")
        if discriminator is not None:
            print(f"Generator  Loss:     {torch.tensor(g_loss).mean():2.6f}")
            print(f"--------------------------------------------------------")
            print(f"D real Loss:  {torch.tensor(d_real_loss).mean():2.6f}")
            print(f"D fake Loss:  {torch.tensor(d_fake_loss).mean():2.6f}")
            print(f"D Loss:       {torch.tensor(d_loss).mean():2.6f}")
        print(f"=======================================================")
        print(f"\n\n\n\n")

        # rec = net.reconstruct(X, train['class_attr'][Y.detach().cpu().numpy()])

        if (ep + 1) == cfg.test_epoch or ((ep + 1) >= cfg.test_epoch and (ep + 1) % cfg.test_period == 0):
            print(f"=======================================================")
            print(f"=========         STARTING  TEST        ===============")
            print(f"=======================================================")

            run_test(generator, train, zsl_test_unseen, gzsl_test_seen, attr_key, test_cfg)















