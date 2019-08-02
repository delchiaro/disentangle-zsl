import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data import get_dataset

class InfiniteDataset(Dataset):
    def __init__(self, len, model, train_feats, train_attrs, train_attrs_bin, test_attrs, **args):
        super().__init__(**args)
        self._len = len
        self._model = model
        self._train_feats = train_feats
        self._train_attrs = train_attrs
        self._train_attrs_bin = train_attrs
        self._test_attrs = test_attrs

        # Extract attirbute embeddings for training set.
        ds = TensorDataset(torch.tensor(train['feats']).float(), torch.tensor(train['labels']).long())
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=6, pin_memory=False, drop_last=False)
        attr_encoding = []
        cntx_encoding = []
        for X in dl:
            X = X[0].to(self._model.device)
            ae, ce = self._model.encode(X)
            attr_encoding.append(ae.detach().cpu())
            cntx_encoding.append(ce.detach().cpu())
        attr_encoding = torch.cat(attr_encoding)
        self._cntx_encoding = torch.cat(cntx_encoding)
        self._attr_encoding = attr_encoding.view([attr_encoding.shape[0],
                                                  self._model.nb_attr,
                                                  self._model.attr_enc_dim])

        # Find valid examples for each attribute.
        valid = []
        for att in range(self._model.nb_attr):
            valid.append(np.where(train_attrs_bin[:, att] == 1)[0])
        self._valid = valid
        
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        cls = np.random.randint(len(self._test_attrs))
        attrs, = np.where(self._test_attrs[cls])
        ret = np.zeros_like(self._attr_encoding[0])
        for idx in attrs:
            emb = self._valid[idx][np.random.randint(len(self._valid[idx]))]
            ret[idx, :] = self._attr_encoding[emb, idx, :]

        return (ret, cls)

if __name__ == '__main__':
    net = torch.load('checkpoint.pt')
    train, val, test_unseen, test_seen = get_dataset('AWA2', use_valid=False, gzsl=False)
    ds = InfiniteDataset(100,
                         net,
                         train['feats'],
                         train['attr'],
                         train['attr_bin'],
                         test_unseen['class_attr_bin'])

#    import matplotlib.pyplot as plt
#    from functools import reduce
#    from textwrap import wrap
#    lens = [int(len(x)) for x in ds._valid]
#    num = reduce(lambda x,y : x*max(1,y), lens, 1)
#    plt.bar(range(1, len(lens)+1), lens)
#    plt.title('\n'.join(wrap(f'# training samples = {num}')))
#    plt.show()

#    nums = []
#    for att in ds._test_attrs:
#        valids = [ds._valid[i] for i in np.where(att)[0]]
#        nums.append(reduce(lambda x,y: x*max(y,1), map(len, valids), 1))
