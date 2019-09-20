import os

from torch.utils.data import DataLoader, TensorDataset

from data import get_dataset, normalize_dataset
import torch
from torch import nn

from disentangle.dataset_generator import FrankensteinDataset, InfiniteDataset
from disentangle.layers import GradientReversal
from exp_5 import Autoencoder, Classifier
from utils import init

#%%
def show_models():
    states_files = sorted(os.listdir('states/'))
    best_state = torch.load(f'states/best_models.pt')
    best_acc = best_state['adapt_acc']
    for state_f in states_files:
        state = torch.load(f'states/{state_f}')
        acc = state['adapt_acc']
        try:
            ep = int(state_f.split('_')[-1].split('.')[0])
            print(f"epoch {ep:03d}  -   acc={acc:2.4f}" + (' --- BEST ' if acc == best_acc else ''))
        except:
            continue


#%%
DATASET = 'AWA2'  # 'CUB'
use_valid = False
attrs_key='class_attr'
mask_attrs_key='class_attr'

device = init(gpu_index=0, seed=42)

train, val, test_unseen, test_seen = get_dataset(DATASET, use_valid=use_valid, gzsl=False, mean_sub=False, std_norm=False, l2_norm=False)
train, val, test_unseen, test_seen = normalize_dataset(train, val, test_unseen, test_seen, keys=('class_attr',), feats_range=(0, 1))
feats_dim = len(train['feats'][0])
nb_train_classes, nb_attributes = train[attrs_key].shape
nb_test_classes = test_unseen[attrs_key].shape[0]

ka_dim = 32
kc_dim = 256  # best: 256, 512, 768
z_dim = 2048
autoencoder = Autoencoder(feats_dim, z_dim, ka_dim, kc_dim, nb_attributes).to(device)
cntx_classifier = nn.Sequential(GradientReversal(1), Classifier(kc_dim, nb_train_classes, )).to(device)
classifier = Classifier(feats_dim, nb_train_classes, (2048,)).to(device)
test_classifier = Classifier(feats_dim, nb_train_classes, (2048,)).to(device)
test_classifier.reset_linear_classifier(nb_test_classes)

def load_models(epoch=None):
    if epoch is None:
        state = torch.load(f'states/best_models.pt')
    else:
        state = torch.load(f'states/models_epoch_{epoch:03d}.pt')
    autoencoder.load_state_dict(state['autoencoder'])
    classifier.load_state_dict(state['classifier'])
    cntx_classifier.load_state_dict(state['cntx_classifier'])
    test_classifier.load_state_dict(state['test_classifier'])


#%%
def show_tsne(model=None, nb_pca=None):
    load_models(model)
    nb_gen_class_samples = 200
    enc_fn = autoencoder.encode

    test_A = torch.tensor(test_unseen[attrs_key]).float().to(device)
    test_A_mask = torch.tensor(test_unseen[mask_attrs_key]).float().to(device)

    test_x = torch.tensor(test_unseen['feats']).float()
    test_y = torch.tensor(test_unseen['labels']).long()
    #test_dataset = TensorDataset(test_x, test_y)
    #test_loader = DataLoader(test_dataset, batch_size=128, num_workers=0)

    # frankenstain_dataset = InfiniteDataset(nb_gen_class_samples, enc_fn,
    #                                        train['feats'], train['labels'], train['attr_bin'], test_unseen['class_attr_bin'], device=device)
    frankenstain_dataset = FrankensteinDataset(nb_gen_class_samples, enc_fn, train['feats'], train['labels'], train['attr_bin'],
                                               test_unseen['class_attr_bin'], device=device)
    frankenstain_loader = DataLoader(frankenstain_dataset, batch_size=128, num_workers=0, shuffle=True)


    X = test_x
    Y = test_y

    X1 = []
    Y1 = []
    for data in frankenstain_loader:
        ka, kc, y = data[0].to(device), data[1].to(device), data[2].to(device)
        a_mask = test_A_mask[y]
        x1 = autoencoder.decode(ka, kc, a_mask)
        X1.append(x1)
        Y1.append(y)
    X1 = torch.cat(X1, dim=0)
    Y1 = torch.cat(Y1, dim=0)

    X, Y, X1, Y1 = (T.detach().cpu().numpy() for T in (X, Y, X1, Y1))


    ### PCA
    if nb_pca is not None:

        from sklearn.decomposition import PCA
        pca = PCA(n_components=nb_pca)
        print("Fitting PCA on real images...")
        pca.fit(X)
        print("Transforming real images with PCA...")
        X = pca.transform(X)
        print("Transforming generated images with PCA...")
        X1 = pca.transform(X1)

    ### TSNE FIT
    print("Fitting TSNE and transforming...")
    import numpy as np
    from tsnecuda import TSNE
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
    embeddings = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(np.concatenate([X, X1], axis=0))
    embeddings1 = embeddings[len(X):]
    embeddings = embeddings[:len(X)]


    # PLOT TSNE
    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(16,10))

    sns.scatterplot(
        x=embeddings[:,0], y=embeddings[:, 1], hue=Y,
        palette=sns.color_palette("hls", 10),
        legend="full",
        alpha=0.6)
    sns.scatterplot(
        x=embeddings1[:,0], y=embeddings1[:, 1], hue=Y1,
        palette=sns.color_palette("hls", 10),
        legend="full",
        alpha=1)
    plt.show()


#%%
show_tsne()
# #%%
# classes = set(Y)
# Xd = {}
# X1d = {}
# for c in classes:
#     idx = np.argwhere(Y==c)[:,0]
#     idx1 = np.argwhere(Y1==c)[:,0]
#     Xd[c] = X[idx]
#     X1d[c] = X1[idx1]
#
#
# for c in classes[0:3]:
#
#     tsne = TSNE(n_components=2).fit(np.concatenate([Xd[c], X1], axis=0), )
#     X_embedded = TSNE(n_components=2).fit_transform(X)
#     X1_embedded = TSNE(n_components=2).fit_transform(X1)