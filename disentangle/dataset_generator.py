import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data import get_dataset
from .net import DisentangleGen, DisentangleEncoder


def FrankensteinDataset(per_class_samples, encoder: DisentangleEncoder, train_feats, train_labels, train_attrs_bin, test_attrs_bin,
                        new_class_offset=0, device=None):
        encoder = encoder
        nb_attributes = train_attrs_bin.shape[1]
        # Extract attirbute embeddings for training set.
        ds = TensorDataset(torch.tensor(train_feats).float(), torch.tensor(train_labels).long())
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=6, pin_memory=False, drop_last=False)
        attr_encoding = []
        cntx_encoding = []
        encoder.to(device)
        for X in dl:
            X = X[0].to(device)
            ae, ce = encoder(X)
            attr_encoding.append(ae.detach().cpu())
            cntx_encoding.append(ce.detach().cpu())
        attr_encoding = torch.cat(attr_encoding)
        cntx_encoding = torch.cat(cntx_encoding)
        attr_encoding = attr_encoding.view([attr_encoding.shape[0], nb_attributes, -1])

        # Find valid examples for each attribute.
        valid = []
        for att in range(nb_attributes):
            valid.append(np.where(train_attrs_bin[:, att] == 1)[0])

        nb_classes =  len(test_attrs_bin)
        frankenstein_attr_enc = np.zeros([per_class_samples*nb_classes, attr_encoding.shape[1], attr_encoding.shape[2]])
        y_trues = []
        for cls in range(nb_classes):
            attr = test_attrs_bin[cls]
            attr_indices, = np.where(attr)
            y_trues += [cls+new_class_offset]*per_class_samples
            for idx in attr_indices:
                try:
                    v = np.random.choice(valid[idx], per_class_samples)
                except ValueError as t:
                    continue# In this case there are no examples with this attribute
                frankenstein_attr_enc[cls*per_class_samples:(cls+1)*per_class_samples, idx, :] = attr_encoding[v, idx, :]

        frankenstein_attr_enc = frankenstein_attr_enc.reshape(frankenstein_attr_enc.shape[0], -1)
        y_trues =np.stack(y_trues)
        random_cntx_enc = cntx_encoding[np.random.choice(len(cntx_encoding), len(frankenstein_attr_enc))]
        return TensorDataset(torch.tensor(frankenstein_attr_enc).float(),
                             random_cntx_enc.float(),
                             torch.tensor(y_trues).long())


class InfiniteDataset(Dataset):
    def __init__(self, len, encoder_fn, train_feats, train_labels, train_attrs_bin, test_attrs_bin,
                 new_class_offset=0, device=None, **args):
        """
        :param len: dataset length
        :param encoder_fn: function that take a batch of image features X and return two tensor containing attribute_embedding and context_embedding
        :param train_feats:
        :param train_labels:
        :param train_attrs_bin:
        :param test_attrs_bin:
        :param new_class_offset:
        :param device:
        :param args:
        """
        super().__init__(**args)
        self._len = len
        self._encoder = encoder_fn
        self._train_feats = train_feats
        self._train_attrs = train_attrs_bin
        self._test_attrs = test_attrs_bin
        self._new_class_offset = new_class_offset
        self._nb_attributes = train_attrs_bin.shape[1]
        # Extract attirbute embeddings for training set.
        ds = TensorDataset(torch.tensor(train_feats).float(), torch.tensor(train_labels).long())
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=6, pin_memory=False, drop_last=False)
        attr_encoding = []
        cntx_encoding = []
        for X in dl:
            X = X[0].to(device)
            ae, ce = self._encoder(X)
            attr_encoding.append(ae.detach().cpu())
            cntx_encoding.append(ce.detach().cpu())
        attr_encoding = torch.cat(attr_encoding)
        self._cntx_encoding = torch.cat(cntx_encoding)
        self._attr_encoding = attr_encoding.view([attr_encoding.shape[0], self._nb_attributes, -1])

        # Find valid examples for each attribute.
        valid = []
        for att in range(self._nb_attributes):
            valid.append(np.where(train_attrs_bin[:, att] == 1)[0])
        self._valid = valid
        
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        cls = np.random.randint(len(self._test_attrs))
        attr = self._test_attrs[cls]
        attr_indices, = np.where(attr)

        random_cntx_enc = self._cntx_encoding[np.random.randint(len(self._cntx_encoding))]
        frankenstein_attr_enc = np.zeros_like(self._attr_encoding[0])
        for idx in attr_indices:
            try:
                emb = self._valid[idx][np.random.randint(len(self._valid[idx]))]
            except ValueError as t:
                # In this case there are no examples with this attribute
                #t.with_traceback()
                #raise t
                continue
            frankenstein_attr_enc[idx, :] = self._attr_encoding[emb, idx, :]

        frankenstein_attr_enc = frankenstein_attr_enc.reshape(-1)
        return frankenstein_attr_enc, random_cntx_enc, cls+self._new_class_offset

if __name__ == '__main__':
    net = torch.load('checkpoint.pt')
    train, val, test_unseen, test_seen = get_dataset('AWA2', use_valid=False, gzsl=False)
    ds = InfiniteDataset(100,
                         net,
                         train['feats'],
                         train['labels'],
                         train['attr'],
                         train['attr_bin'],
                         test_unseen['class_attr_bin'])

