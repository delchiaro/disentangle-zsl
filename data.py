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
    os.makedirs(DATA_PATH+dataset_name, exist_ok=True)
    xlsa_data_path = str(DATA_PATH+'xlsa17/data/')
    mat_feat = scipy.io.loadmat(xlsa_data_path + '' + dataset_name + '/res101.mat')
    features = mat_feat['features'].T
    labels = mat_feat['labels']
    mat = scipy.io.loadmat(xlsa_data_path + '' + dataset_name + '/att_splits.mat')
    attributes = mat['att'].T
    attributes_orig = mat['original_att'].T

    split_names = ['trainval', 'test_seen', 'test_unseen', 'train', 'val']
    for split in split_names:
        print(split)
        locs = mat[split + '_loc']
        features_temp = np.zeros((locs.shape[0], features.shape[1]))
        labels_temp = np.zeros((locs.shape[0], np.amax(labels)))
        attributes_temp = np.zeros((locs.shape[0], attributes.shape[1]))
        attributes_orig_temp = np.zeros((locs.shape[0], attributes_orig.shape[1]))
        for i, loc in enumerate(locs):
            features_temp[i] = features[loc - 1]
            labels_temp[i, labels[loc - 1] - 1] = 1
            attributes_temp[i] = attributes[labels[loc - 1] - 1]
            attributes_orig_temp[i] = attributes_orig[labels[loc - 1] - 1]
        np.save(DATA_PATH + dataset_name + '/' + dataset_name + '_' + split + '_features', features_temp)
        np.save(DATA_PATH + dataset_name + '/' + dataset_name + '_' + split + '_labels', labels_temp)
        np.save(DATA_PATH + dataset_name + '/' + dataset_name + '_' + split + '_attributes', attributes_temp)
        np.save(DATA_PATH + dataset_name + '/' + dataset_name + '_' + split + '_attributes_orig', attributes_orig_temp)
        np.save(DATA_PATH + dataset_name + '/' + dataset_name + '_class_attributes', attributes)
        np.save(DATA_PATH + dataset_name + '/' + dataset_name + '_class_attributes_orig', attributes_orig)

    print("=======")


def download_preprocess_all():
    download_data()
    data_set = ['APY', 'AWA1', 'AWA2', 'CUB', 'SUN']
    for name in data_set:
        preprocess_dataset(name)
    shutil.rmtree(DATA_PATH+'xlsa17')


def get_data(dataset, split):
    assert dataset in ['APY', 'AWA1', 'AWA2', 'CUB', 'SUN']
    assert split in ['test_seen', 'test_unseen', 'train', 'val', 'trainval']
    data = {}

    data['feats'] = np.load(DATA_PATH+f'{dataset}/{dataset}_{split}_features.npy')
    data['attr'] = np.load(DATA_PATH+f'{dataset}/{dataset}_{split}_attributes.npy')
    data['attr_orig'] = np.load(DATA_PATH+f'{dataset}/{dataset}_{split}_attributes_orig.npy')


    labels_glob = np.argmax(np.load(DATA_PATH+f'{dataset}/{dataset}_{split}_labels.npy'), axis=1)
    glob2loc = {glob_lbl: loc_lbl for loc_lbl, glob_lbl in enumerate(sorted(set(labels_glob)))}
    loc2glob = {l: g for g, l in glob2loc.items()}
    data['labels'] = np.array([glob2loc[glob] for glob in labels_glob])
    data['labels_glob'] = labels_glob
    data['glob2loc'] = glob2loc
    data['loc2glob'] = loc2glob

    # local_classes = sorted(loc2glob.keys())
    # class_attr = np.zeros([len(local_classes), data['attr'].shape[1]])
    # class_attr_orig = np.zeros([len(local_classes), data['attr_orig'].shape[1]])
    # for label in local_classes:
    #     idx = np.argwhere(data['labels'] == label)
    #     class_attr[label] = np.mean(data['attr'][idx], axis=0)
    #     class_attr_orig[label] = np.mean(data['attr_orig'][idx], axis=0)
    # #data['class_attr'] = class_attr
    # #data['class_attr_orig'] = class_attr_orig

    classes_global = sorted(set(data['labels_glob']))
    data['class_attr'] = np.load(DATA_PATH+f'{dataset}/{dataset}_class_attributes.npy')[classes_global]
    data['class_attr_orig'] = np.load(DATA_PATH+f'{dataset}/{dataset}_class_attributes_orig.npy')[classes_global]
    data['class_attr_bin'] = (data['class_attr_orig'] > data['class_attr_orig'].mean()).astype('float')
    data['attr_bin'] = (data['attr_orig'] > data['class_attr_orig'].mean()).astype('float')

    return data


def get_dataset(dataset, use_valid=False, gzsl=False):
    if use_valid:
        train = get_data(dataset, 'train')
        val = get_data(dataset, 'val')
    else:
        train = get_data(dataset, 'trainval')
        val = None

    test_unseen = get_data(dataset, 'test_unseen')
    test_seen = None
    if gzsl:
        test_seen = get_data(dataset, 'test_seen')

    return train, val, test_unseen, test_seen


def normalize_dataset(train, val=None, test=None, feats_range=(0, 1)):
    if feats_range is not None:
        min_max_scaler = preprocessing.MinMaxScaler(feats_range)
        train['feats'] = min_max_scaler.fit_transform(train['feats'])
        if val is not None:
            val['feats'] = min_max_scaler.transform(val['feats'])
        if test is not None:
            test['feats'] = min_max_scaler.transform(test['feats'])
    return train, val, test
