import os

from torch.utils.data import DataLoader, TensorDataset

from data import get_dataset, normalize_dataset
import torch
from torch import nn

from disentangle.dataset_generator import FrankensteinDataset, InfiniteDataset
from disentangle.layers import GradientReversal
from exp_5 import Autoencoder, Classifier, gen_zsl_test, Model, init_exp, build_model_A
from utils import init
from sklearn.decomposition import PCA
import numpy as np
from tsnecuda import TSNE
import seaborn as sns
from matplotlib import pyplot as plt


def tsne(model: Model, test_dicts, train_dict, nb_pca=None, nb_gen_class_samples=200, infinite_dataset=False,
         target='feats', # 'attr_emb', 'cntx_emb'
         append_title='',
         attrs_key='class_attr', mask_attrs_key='class_attr',
         savepath=None, dpi=250):
    test_dicts = [test_dicts] if isinstance(test_dicts, dict) else test_dicts

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(nrows=len(test_dicts), figsize=(20, 8 * len(test_dicts)), dpi=dpi)
    title = f'tsne {target}{append_title}'
    fig.suptitle(title, fontsize=16)

    for i, data_dict in enumerate(test_dicts):
        nb_classes = len(set(data_dict['labels']))
        test_A = torch.tensor(data_dict[attrs_key]).float().to(model.device)
        test_A_mask = torch.tensor(data_dict[mask_attrs_key]).float().to(model.device)
        X = torch.tensor(data_dict['feats']).float()
        Y = torch.tensor(data_dict['labels']).long()
        dataset = TensorDataset(X, Y)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

        if infinite_dataset:
            frankenstain_dataset = InfiniteDataset(nb_gen_class_samples*nb_classes, model.autoencoder.encode,
                                                   train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                                   data_dict['class_attr_bin'], device=model.device)
        else:
            frankenstain_dataset = FrankensteinDataset(nb_gen_class_samples, model.autoencoder.encode,
                                                       train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
                                                       data_dict['class_attr_bin'], device=model.device)
        frankenstain_loader = DataLoader(frankenstain_dataset, batch_size=128, num_workers=0, shuffle=True)


        if target in ['cntx_emb', 'attr_emb']:
            KA1 = []; KC1 = []; Y1 = []
            for ka, kc, y in frankenstain_loader:
                ka = model.autoencoder.mask_ka(ka.to(model.device), test_A_mask[y.to(model.device)])
                KA1.append(ka);  KC1.append(kc); Y1.append(y)

            KA = []; KC = []; Y = []
            for x, y in data_loader:
                ka, kc = model.autoencoder.encode(x.to(model.device))
                ka = model.autoencoder.mask_ka(ka, test_A_mask[y.to(model.device)])
                KA.append(ka);
                KC.append(kc);
                Y.append(y)
            KA, KC, Y, KA1, KC1, Y1 = (torch.cat(T, dim=0).detach().cpu().numpy() for T in (KA, KC, Y, KA1, KC1, Y1) )
            if target == 'cntx_emb':
                X, X1 = KC, KC1
            elif target is 'attr_emb':
                X, X1 = KA, KA1

        else:
            X1 = []
            Y1 = []
            for data in frankenstain_loader:
                ka, kc, y = data[0].to(model.device), data[1].to(model.device), data[2].to(model.device)
                x1 = model.autoencoder.decode(ka, kc, test_A_mask[y])
                X1.append(x1); Y1.append(y)
            X1 = torch.cat(X1, dim=0); Y1 = torch.cat(Y1, dim=0)
            X, Y, X1, Y1 = (T.detach().cpu().numpy() for T in (X, Y, X1, Y1))


        ### PCA
        if nb_pca is not None:
            pca = PCA(n_components=nb_pca)
            print("Fitting PCA on real images...")
            pca.fit(X)
            print("Transforming real images with PCA...")
            X = pca.transform(X)
            print("Transforming generated images with PCA...")
            X1 = pca.transform(X1)

        ### TSNE FIT
        print("Fitting TSNE and transforming...")
        embeddings = TSNE(n_components=2).fit_transform(np.concatenate([X, X1], axis=0))
        embeddings1 = embeddings[len(X):]
        embeddings = embeddings[:len(X)]

        # PLOT TSNE
        from matplotlib import pyplot as plt
        import seaborn as sns
        sns.scatterplot(x=embeddings[:,0], y=embeddings[:, 1], hue=Y,
                        palette=sns.color_palette("hls", nb_classes), legend=False, alpha=0.4, ax=axs[i])
        sns.scatterplot(x=embeddings1[:,0], y=embeddings1[:, 1], hue=Y1,
                        palette=sns.color_palette("hls", nb_classes), legend=False, alpha=1, ax=axs[i])
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, title+'.png'))
    else:
        plt.show()


#%%


def main():
    model, train, test_unseen, test_seen, val = init_exp(build_model_A, 0, 42, use_valid=False)
    savepath = None #'tsne/exp_5'

    for i in range(0, 4, 2):
        model.load(state_dir='states/exp_5', epoch=i)
        app_title = f'- ep={i} - acc={model.acc*100:2.2f}'
        tsne(model, [test_unseen, train], train, infinite_dataset=True, append_title=app_title, savepath=savepath, target='feats')
        tsne(model, [test_unseen, train], train, infinite_dataset=True, append_title=app_title, savepath=savepath, target='attr_emb')

if __name__ == '__main__':
    main()
#%%
# Please cite the corresponding paper if it was useful for your research:
#
# @article{chan2019gpu,
#   title={GPU accelerated t-distributed stochastic neighbor embedding},
#   author={Chan, David M and Rao, Roshan and Huang, Forrest and Canny, John F},
#   journal={Journal of Parallel and Distributed Computing},
#   volume={131},
#   pages={1--13},
#   year={2019},
#   publisher={Elsevier}
# }


#
# def tsne_on_enc(model: Model, test_dicts, train_dict, nb_pca=None, nb_gen_class_samples=200, infinite_dataset=False, show_context=False,
#                 append_title='',
#                 attrs_key='class_attr', mask_attrs_key='class_attr',
#                 savepath=None, dpi=250):
#     if isinstance(test_dicts, dict):
#         test_dicts = [test_dicts]
#     fig, axs = plt.subplots(nrows=len(test_dicts), figsize=(20, 8 * len(test_dicts)), dpi=dpi)
#     title = ('tsne encoded-context' if show_context else 'tsne encoded-attributes') + append_title
#     fig.suptitle(title, fontsize=16)
#
#     for i, data_dict in enumerate(test_dicts):
#         nb_classes = len(set(data_dict['labels']))
#
#         test_A = torch.tensor(data_dict[attrs_key]).float().to(model.device)
#         test_A_mask = torch.tensor(data_dict[mask_attrs_key]).float().to(model.device)
#
#         X = torch.tensor(data_dict['feats']).float()
#         Y = torch.tensor(data_dict['labels']).long()
#         dataset = TensorDataset(X, Y)
#         data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
#
#         if infinite_dataset:
#             frankenstain_dataset = InfiniteDataset(nb_gen_class_samples*nb_classes, model.autoencoder.encode,
#                                                    train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
#                                                    data_dict['class_attr_bin'], device=model.device)
#         else:
#             frankenstain_dataset = FrankensteinDataset(nb_gen_class_samples, model.autoencoder.encode,
#                                                        train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
#                                                        data_dict['class_attr_bin'], device=model.device)
#         frankenstain_loader = DataLoader(frankenstain_dataset, batch_size=128, num_workers=0, shuffle=True)
#
#         KA1 = []; KC1 = []; Y1 = []
#         for ka, kc, y in frankenstain_loader:
#             ka = model.autoencoder.mask_ka(ka.to(model.device), test_A_mask[y.to(model.device)])
#             KA1.append(ka);  KC1.append(kc); Y1.append(y)
#
#         KA = []; KC = []; Y = []
#         for x, y in data_loader:
#             ka, kc = model.autoencoder.encode(x.to(model.device))
#             ka = model.autoencoder.mask_ka(ka, test_A_mask[y.to(model.device)])
#             KA.append(ka);
#             KC.append(kc);
#             Y.append(y)
#         KA, KC, Y, KA1, KC1, Y1 = (torch.cat(T, dim=0).detach().cpu().numpy() for T in (KA, KC, Y, KA1, KC1, Y1) )
#
#         if show_context:
#             X, X1 = KC, KC1
#         else:
#             X, X1 = KA, KA1
#
#         ### PCA
#         if nb_pca is not None:
#             pca = PCA(n_components=nb_pca)
#             print("Fitting PCA on real images...")
#             pca.fit(X)
#             print("Transforming real images with PCA...")
#             X = pca.transform(X)
#             print("Transforming generated images with PCA...")
#             X1 = pca.transform(X1)
#
#         ### TSNE FIT
#         print("Fitting TSNE and transforming...")
#         embeddings = TSNE(n_components=2).fit_transform(np.concatenate([X, X1], axis=0))
#         embeddings1 = embeddings[len(X):]
#         embeddings = embeddings[:len(X)]
#
#         # PLOT TSNE
#         from matplotlib import pyplot as plt
#         import seaborn as sns
#         sns.scatterplot(x=embeddings[:,0], y=embeddings[:, 1], hue=Y,
#                         palette=sns.color_palette("hls", nb_classes), legend=False, alpha=0.4, ax=axs[i])
#         sns.scatterplot(x=embeddings1[:,0], y=embeddings1[:, 1], hue=Y1,
#                         palette=sns.color_palette("hls", nb_classes), legend=False, alpha=1, ax=axs[i])
#     if savepath is not None:
#         plt.savefig(os.path.join(savepath, title+'.png'))
#     else:
#         plt.show()
#
# def tsne_on_feats(model: Model, test_dict, train_dict, nb_pca=None, nb_gen_class_samples=400, infinite_dataset=False,
#                   append_title='', attrs_key='class_attr', mask_attrs_key='class_attr',
#                   savepath=None, dpi=250):
#     if isinstance(test_dict, dict):
#         test_dict = [test_dict]
#
#     title = 'TSNE encoded features' + append_title
#     fig, axs = plt.subplots(nrows=len(test_dict), figsize=(20, 8 * len(test_dict)), dpi=dpi)
#     fig.suptitle(title, fontsize=16)
#
#     for i, data_dict in enumerate(test_dict):
#         nb_classes = len(set(data_dict['labels']))
#         test_A = torch.tensor(data_dict[attrs_key]).float().to(model.device)
#         test_A_mask = torch.tensor(data_dict[mask_attrs_key]).float().to(model.device)
#
#         X = torch.tensor(data_dict['feats']).float()
#         Y = torch.tensor(data_dict['labels']).long()
#
#         if infinite_dataset:
#             frankenstain_dataset = InfiniteDataset(nb_gen_class_samples * nb_classes, model.autoencoder.encode, train_dict['feats'],
#                                                    train_dict['labels'], train_dict['attr_bin'],
#                                                    data_dict['class_attr_bin'], device=model.device)
#         else:
#             frankenstain_dataset = FrankensteinDataset(nb_gen_class_samples, model.autoencoder.encode, train_dict['feats'], train_dict['labels'], train_dict['attr_bin'],
#                                                        data_dict['class_attr_bin'], device=model.device)
#         frankenstain_loader = DataLoader(frankenstain_dataset, batch_size=128, num_workers=0, shuffle=True)
#
#
#         X1 = []
#         Y1 = []
#         for data in frankenstain_loader:
#             ka, kc, y = data[0].to(model.device), data[1].to(model.device), data[2].to(model.device)
#             x1 = model.autoencoder.decode(ka, kc, test_A_mask[y])
#             X1.append(x1); Y1.append(y)
#         X1 = torch.cat(X1, dim=0); Y1 = torch.cat(Y1, dim=0)
#         X, Y, X1, Y1 = (T.detach().cpu().numpy() for T in (X, Y, X1, Y1))
#
#         ### PCA
#         if nb_pca is not None:
#             pca = PCA(n_components=nb_pca)
#             print("Fitting PCA on real images...")
#             pca.fit(X)
#             print("Transforming real images with PCA...")
#             X = pca.transform(X)
#             print("Transforming generated images with PCA...")
#             X1 = pca.transform(X1)
#
#         ### TSNE FIT
#         print("Fitting TSNE and transforming...")
#         embeddings = TSNE(n_components=2).fit_transform(np.concatenate([X, X1], axis=0))
#         embeddings1 = embeddings[len(X):]
#         embeddings = embeddings[:len(X)]
#
#         # PLOT TSNE
#         sns.scatterplot(x=embeddings[:,0], y=embeddings[:, 1], hue=Y,
#                         palette=sns.color_palette("hls", nb_classes), legend=False, alpha=0.4, ax=axs[i])
#         sns.scatterplot(x=embeddings1[:,0], y=embeddings1[:, 1], hue=Y1,
#                         palette=sns.color_palette("hls", nb_classes), legend=False, alpha=1, ax=axs[i])
#     if savepath is not None:
#         plt.savefig(os.path.join(savepath, title+'.png'))
#     else:
#         plt.show()
#
