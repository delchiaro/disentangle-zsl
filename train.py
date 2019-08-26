
import torch.nn.functional as NN
from progressbar import progressbar
import torch

from disentangle.net import generate_frankenstain_attr_enc
from test import run_test
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def run_train(net, opt,
              train, val, test_unseen, test_seen,
              nb_epochs=100, bs=128,
              discriminator=None, discr_opt=None,
              device=None,
              first_test_epoch=1, test_period=1,
              nb_class_epochs=5,
              adapt_lr=None,
              seen_samples_mean=8,
              seen_samples_std=3,
              attr_key='class_attr',
              use_infinite_dataset=False,
              nb_gen_class_samples=200):

    train_dataset = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6, pin_memory=False, drop_last=False)
    nb_train_classes = len(set(train['labels']))


    A_train_all = torch.tensor(train[attr_key]).float().to(device)
    for ep in range(nb_epochs):
        reconstruction_loss = torch.zeros(len(train_loader))
        attribute_rec_loss = torch.zeros(len(train_loader))
        classification_loss = torch.zeros(len(train_loader))
        context_class_loss = torch.zeros(len(train_loader))
        disentangling_loss = torch.zeros(len(train_loader))
        g_discriminator_loss = torch.zeros(len(train_loader))
        d_real_loss = torch.zeros(len(train_loader))
        d_fake_loss = torch.zeros(len(train_loader))
        for bi, (X, Y) in enumerate(progressbar(train_loader)):
            X = X.to(device)
            Y = Y.to(device)
            attr = A_train_all[Y]
            zeros = torch.zeros([X.shape[0], 1]).to(net.device)
            ones = torch.ones([X.shape[0], 1]).to(net.device)

            #### TRAINING GENERATOR NET
            opt.zero_grad()
            # Forward
            decoded, logits, cntx_logits, (attr_enc, cntx_enc) = net.forward(X, attr)
            attr_reconstr = net.reconstruct_attributes(attr_enc)

            # Computing Losses
            l_reconstr = NN.l1_loss(decoded, X  )  # *X.shape[1]  # NN.mse_loss(decoded, X)
            l_attr_reconstr = NN.l1_loss(attr_reconstr, attr  )  # *attr.shape[1]
            # attr_reconstruct_loss = torch.Tensor([0.]).to(self.device)
            l_class = NN.cross_entropy(logits, Y)
            l_cntx_class = NN.cross_entropy(cntx_logits, Y)
            l_disentangle = net.disentangle_loss(attr_enc, cntx_enc, Y, A_train_all)
            l_discr = 0

            # Computing Discriminator Fooling Loss
            if discriminator is not None:
                frnk_attr_enc = generate_frankenstain_attr_enc(net, attr_enc)
                l_discr = NN.binary_cross_entropy_with_logits(discriminator(decoded), zeros)

            # loss = 1*l_reconstr + 1*l_attr_reconstr + 1*l_class + 100*l_cntx_class + 2*l_disentangle + 5*l_discr
            loss = 100 *l_reconstr + 1* l_attr_reconstr + 1 * l_class + 2 * l_cntx_class + 2 * l_disentangle + 5 * l_discr
            loss.backward()
            opt.step()
            # attr_enc = attr_enc.detach()
            decoded = decoded.detach()

            #### TRAINING DISCRIMINATOR NET
            if discriminator is not None:
                discr_opt.zero_grad()
                zeros = torch.zeros([X.shape[0], 1]).to(net.device)
                ones = torch.ones([X.shape[0], 1]).to(net.device)
                l_discr_real = NN.binary_cross_entropy_with_logits(discriminator(X), zeros)
                l_discr_fake = NN.binary_cross_entropy_with_logits(discriminator(decoded), ones)
                dloss = (l_discr_real + l_discr_fake) / 2
                dloss.backward()
                discr_opt.step()
                discr_opt.zero_grad()
                d_real_loss[bi] = l_discr_real
                d_fake_loss[bi] = l_discr_fake

            reconstruction_loss[bi] = l_reconstr
            attribute_rec_loss[bi] = l_attr_reconstr
            classification_loss[bi] = l_class
            context_class_loss[bi] = l_cntx_class
            disentangling_loss[bi] = l_disentangle
            g_discriminator_loss[bi] = l_discr

        print(f"")
        print(f"=======================================================")
        print(f"Epoch {ep + 1}/{nb_epochs} ended: ")
        print(f"=======================================================")
        print(f"Reconstruction Loss: {torch.mean(reconstruction_loss):2.6f}")
        print(f"Attr Reconstr  Loss: {torch.mean(attribute_rec_loss):2.6f}")
        print(f"Classification Loss: {torch.mean(classification_loss):2.6f}")
        print(f"Context Class  Loss: {torch.mean(context_class_loss):2.6f}")
        print(f"Disentangling  Loss: {torch.mean(disentangling_loss):2.6f}")
        print(f"Discriminator  Loss: {torch.mean(g_discriminator_loss):2.6f}")
        print(f"--------------------------------------------------------")
        print(f"D real Loss:  {torch.mean(d_real_loss):2.6f}")
        print(f"D fake Loss:  {torch.mean(d_fake_loss):2.6f}")
        print(f"=======================================================")
        print(f"\n\n\n\n")

        # rec = net.reconstruct(X, train['class_attr'][Y.detach().cpu().numpy()])

        if (ep + 1) == first_test_epoch or ((ep + 1) >= first_test_epoch and (ep + 1) % test_period == 0):
            print(f"=======================================================")
            print(f"=========         STARTING  TEST        ===============")
            print(f"=======================================================")

            run_test(net, train, test_unseen, test_seen,
                     nb_epochs=nb_class_epochs, perclass_gen_samples=nb_gen_class_samples,
                     threshold=float(np.mean(test_unseen[attr_key])),
                     use_infinite_dataset=use_infinite_dataset,
                     seen_samples_mean=seen_samples_mean, seen_samples_std=seen_samples_std,
                     lr=adapt_lr)
