from pathlib import Path

from sklearn import preprocessing
import numpy as np
import scipy.io
import os
import shutil

DATA_PATH = './data/'

def download_data():
    os.mkdir('data')
    os.system('wget http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip')
    os.system('unzip ./data/xlsa17.zip')
    os.remove(DATA_PATH+'xlsa17.zip')


def preprocess_dataset(dataset_name):
    print(dataset_name)
    os.makedirs(dataset_name)
    path = str(DATA_PATH+'xlsa17/data/')
    mat_feat = scipy.io.loadmat(path + '' + dataset_name + '/res101.mat')
    features = mat_feat['features'].T
    labels = mat_feat['labels']
    mat = scipy.io.loadmat(path + '' + dataset_name + '/att_splits.mat')
    attributes = mat['att'].T
    split_name = ['trainval', 'test_seen', 'test_unseen', 'train', 'val']
    for name in split_name:
        print(name)
        locs = mat[name + '_loc']
        features_temp = np.zeros((locs.shape[0], features.shape[1]))
        labels_temp = np.zeros((locs.shape[0], np.amax(labels)))
        attributes_temp = np.zeros((locs.shape[0], attributes.shape[1]))
        for i, loc in enumerate(locs):
            features_temp[i] = features[loc - 1]
            labels_temp[i, labels[loc - 1] - 1] = 1
            attributes_temp[i] = attributes[labels[loc - 1] - 1]
        np.save(dataset_name + '/' + dataset_name + '_' + name + '_features', features_temp)
        np.save(dataset_name + '/' + dataset_name + '_' + name + '_labels', labels_temp)
        np.save(dataset_name + '/' + dataset_name + '_' + name + '_attributes', attributes_temp)
    print("=======")


def download_preprocess_all():
    download_data()
    data_set = ['APY', 'AWA1', 'AWA2', 'CUB', 'SUN']
    for name in data_set:
        preprocess_dataset(name)
    shutil.rmtree(DATA_PATH+'xlsa17')


def get_data(dataset, split):
    assert dataset in ['APY', 'AWA', 'AWA2', 'CUB', 'SUN']
    assert split in ['test_seen', 'test_unseen', 'train', 'val', 'trainval']
    data = {}

    data['feats'] = np.load(DATA_PATH+f'{dataset}/{dataset}_{split}_features.npy')
    data['attr'] = np.load(DATA_PATH+f'{dataset}/{dataset}_{split}_attributes.npy')

    labels_glob = np.argmax(np.load(DATA_PATH+f'{dataset}/{dataset}_{split}_labels.npy'), axis=1)
    glob2loc = {glob_lbl: loc_lbl for loc_lbl, glob_lbl in enumerate(sorted(set(labels_glob)))}
    loc2glob = {l: g for g, l in glob2loc.items()}
    data['labels'] = np.array([glob2loc[glob] for glob in labels_glob])
    data['labels_glob'] = labels_glob
    data['glob2loc'] = glob2loc
    data['loc2glob'] = loc2glob

    local_classes = sorted(loc2glob.keys())
    class_attr = np.zeros([len(local_classes), data['attr'].shape[1]])
    for label in local_classes:
        idx = np.argwhere(data['labels'] == label)
        class_attr[label] = np.mean(data['attr'][idx], axis=0)
    data['class_attr'] = class_attr
    return data


def get_dataset(dataset, use_valid=False, gzsl=False):
    if use_valid:
        train = get_data(dataset, 'train')
        val = get_data(dataset, 'val')
    else:
        train = get_data(dataset, 'trainval')
        val = None

    test = get_data(dataset, 'test_unseen')
    if gzsl:
        test_seen = get_data(dataset, 'test_seen')
        test = [np.concatenate([unseen_stuff, seen_stuff], axis=0) for unseen_stuff, seen_stuff in zip(test, test_seen)]

    return train, val, test


def normalize_dataset(train, val=None, test=None, feats_range=(0, 1)):
    if feats_range is not None:
        min_max_scaler = preprocessing.MinMaxScaler(feats_range)
        train['feats'] = min_max_scaler.fit_transform(train['feats'])
        if val is not None:
            val['feats'] = min_max_scaler.transform(val['feats'])
        if test is not None:
            test['feats'] = min_max_scaler.transform(test['feats'])
    return train, val, test
