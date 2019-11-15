import numpy as np
import torch
import random

from typing import Tuple

from sklearn.decomposition import PCA
from tsnecuda import TSNE


def NP(tensor):
    return tensor.detach().cpu().numpy()

def unison_shuffled_copies(list_of_arrays):
    for a in list_of_arrays:
        assert len(a) == len(list_of_arrays[0])
    p = np.random.permutation(len(list_of_arrays[0]))
    return (a[p] for a in list_of_arrays)


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical




def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def init(gpu_index=None, seed=None, verbose=True):
    gpu_index = gpu_index if gpu_index is not None else -1
    if seed is not None:
        set_seed(seed)
    if gpu_index >= 0:
        torch.cuda.set_device(gpu_index)
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')

    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() and gpu_index >= 0 else "cpu")
    if verbose:
        print("Torch will use device: {}".format(device))
    return device



import os

def tsne_plot(data: Tuple[torch.FloatTensor], markers=['o', 'X', '.'], alphas=[1, 0.5, 0.3],
              nb_pca_components=None, append_title='', legend=False, savepath=None, figsize=(14, 8),  dpi=250,
              **kwargs):

    if isinstance(data[0], torch.Tensor):
        data = [data]
    Xs = [NP(d[0]) for d in data]
    X = np.concatenate(Xs, axis=0)

    Ys = [NP(d[1]) for d in data]
    Y =  np.concatenate(Ys, axis=0)
    ### PCA
    if nb_pca_components is not None:
        pca = PCA(n_components=nb_pca_components)
        print("Fitting PCA and transforming...")
        pca.fit(X)
        X = pca.transform(X)

    ### TSNE FIT
    print("Fitting TSNE and transforming...")
    tsne_data = TSNE(n_components=2, **kwargs).fit_transform(X)

    b = len(Xs[0])
    Ts = [tsne_data[:b]]
    for x in (Xs[1:]):
        new_b = b+len(x)
        Ts.append(tsne_data[b:new_b])
        b = new_b

    # PLOT TSNE
    from matplotlib import pyplot as plt
    import seaborn as sns
    fig = plt.figure(figsize=figsize, dpi=dpi)
    title = f'tsne {append_title}'
    fig.suptitle(title, fontsize=16)
    for T, Y, m, a in zip(Ts, Ys, markers[:len(Ts)], alphas[:len(Ts)]):
        sns.scatterplot(x=T[:, 0], y=T[:, 1], hue=Y[:, 0], marker=m, alpha=a, palette=sns.color_palette("hls", len(set(Y[:,0]))),
                        legend=legend)

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, title + '.png'))
    else:
        plt.show()
