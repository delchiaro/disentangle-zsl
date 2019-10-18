import torch
import numpy as np
from scipy.special import softmax
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data import get_dataset
from scipy.spatial.distance import cdist
from .net import DisentangleGen, DisentangleEncoder


class InfiniteDataset(Dataset):
    def __init__(self, length, encoder_fn, train_feats, train_labels, train_attrs_bin, test_attrs_bin,
                 new_class_offset=0, device=None, use_context=True, **args):
        """
        :param length: dataset length
        :param encoder_fn: function that take a batch of image features X and return two tensor containing attribute_embedding and
        context_embedding
        :param train_feats:
        :param train_labels:
        :param train_attrs_bin:
        :param test_attrs_bin:
        :param new_class_offset:
        :param device:
        :param args:
        """
        super().__init__(**args)
        self._len = length
        self._encoder = encoder_fn
        self._train_feats = train_feats
        self._train_attrs = train_attrs_bin
        self._test_attrs = test_attrs_bin
        self._new_class_offset = new_class_offset
        self._nb_attributes = train_attrs_bin.shape[1]
        self._device = device
        # Extract attirbute embeddings for training set.
        ds = TensorDataset(torch.tensor(train_feats).float(), torch.tensor(train_labels).long())
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=6, pin_memory=False, drop_last=False)


        enc = self._encoder(next(iter(dl))[0].to(device))
        if isinstance(enc, tuple) or isinstance(enc, set) or isinstance(enc, list):
            encodings = [[]]* len(enc)
            def add_fn(tensors):
                for t, E in zip(tensors, encodings):
                    E.append(t.detach().cpu())
        else:
            encodings = [[]]
            def add_fn(tensor):
                encodings[0].append(tensor.detach().cpu())

        for X in dl:
            X = X[0].to(device)
            tensors =  self._encoder(X)
            add_fn(tensors)
        encodings = [torch.cat(E, dim=0) for E in encodings]
        self.encodings = [E.view(E.shape[0], self._nb_attributes, -1) for E in encodings]

        # Find valid examples for each attribute.
        idx_active_attr = []
        train_examples_attrs_bin = train_attrs_bin[train_labels]
        for att in range(self._nb_attributes):
            idx_active_attr.append(np.where(train_examples_attrs_bin[:, att] == 1)[0])
        self._idx_active_attr = idx_active_attr

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        cls = np.random.randint(len(self._test_attrs))
        return self.get_item_with_class(cls)

    def get_items_with_class(self, Y):
        ENC, CLS = [], []
        for cls in Y:
            enc, y = self.get_item_with_class(cls)
            #AE.append(ae[None, :]); CE.append(ce[None, :]), CLS.append(y)
            ENC.append(enc[None, :]); CLS.append(y)
        return  torch.cat(ENC),  torch.tensor(CLS)


    def get_item_with_class(self, cls):
        attr = self._test_attrs[cls]
        attr_indices, = np.where(attr)

        frnk_enc = [np.zeros_like(E[0]).reshape([self._nb_attributes, -1]) for E in self.encodings]

        for idx in attr_indices:
            try:
                emb = self._idx_active_attr[idx][np.random.randint(len(self._idx_active_attr[idx]))]
            except ValueError as t:
                # In this case there are no  examples with this attribute
                # t.with_traceback()
                # raise t
                continue
            for i in range(len(self.encodings)):
                frnk_enc[i][idx, :] = self.encodings[i][emb, idx, :]

        for i in range(len(self.encodings)):
            frnk_enc[i] = torch.tensor(frnk_enc[i].reshape(-1))
        cls = cls + self._new_class_offset
        return frnk_enc, torch.tensor(cls)




class ProbabilisticInfiniteDataset(Dataset):
    def __init__(self, length, encoder_fn, train_feats, train_labels, train_attrs_bin, test_attrs_bin,
                 new_class_offset=0, device=None, use_context=True, **args):
        """
        :param length: dataset length
        :param encoder_fn: function that take a batch of image features X and return two tensor containing attribute_embedding and
        context_embedding
        :param train_feats:
        :param train_labels:
        :param train_attrs_bin:
        :param test_attrs_bin:
        :param new_class_offset:
        :param device:
        :param args:
        """
        super().__init__(**args)
        self._len = length
        self._encoder = encoder_fn
        self._train_feats = train_feats
        self._train_attrs = train_attrs_bin
        self._test_attrs = test_attrs_bin
        self._new_class_offset = new_class_offset
        self._nb_attributes = train_attrs_bin.shape[1]
        self._device = device
        # Extract attirbute embeddings for training set.
        ds = TensorDataset(torch.tensor(train_feats).float(), torch.tensor(train_labels).long())
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=6, pin_memory=False, drop_last=False)


        enc = self._encoder(next(iter(dl))[0].to(device))
        if isinstance(enc, tuple) or isinstance(enc, set) or isinstance(enc, list):
            encodings = [[]]* len(enc)
            def add_fn(tensors):
                for t, E in zip(tensors, encodings):
                    E.append(t.detach().cpu())
        else:
            encodings = [[]]
            def add_fn(tensor):
                encodings[0].append(tensor.detach().cpu())

        for X in dl:
            X = X[0].to(device)
            tensors =  self._encoder(X)
            add_fn(tensors)
        encodings = [torch.cat(E, dim=0) for E in encodings]
        self.encodings = [E.view(E.shape[0], self._nb_attributes, -1) for E in encodings]

        # Find valid examples for each attribute.
        train_examples_attrs_bin = train_attrs_bin[train_labels]
        idx_active_attr = []
        for att in range(self._nb_attributes):
            idx_active_attr.append(np.where(train_examples_attrs_bin[:, att] == 1)[0])

        idx_with_label = []
        for label in sorted(set(train_labels)):
            idx_with_label.append(np.where(train_labels == label)[0])


        D = cdist([av for av in test_attrs_bin], [av for av in train_attrs_bin], 'cityblock')
        self.class_attribute_similarity = self._nb_attributes - D
        if np.all(train_attrs_bin == train_attrs_bin):
            np.fill_diagonal(self.class_attribute_similarity, 0)
        #self.probs = softmax(S, axis=1)


        # Precompute all the similarities (and so - probabilities) of each test-class to all the train classes based on
        # attribute vector distances.
        # We do this for each test-class, and we repeat this for each attribute because we want to mask/prune the similarities
        # with train-classes not having that attribute active.
        P = []
        for test_cls in range(len(test_attrs_bin)):
            attr_to_train_cls_probs = []
            for attr_id in range(self._nb_attributes):
                # Create a mask with lenght equal to the number of training classes,
                # having 1 for classes having the current attribute 'attr_id" active.
                classes_mask = np.asarray(self._train_attrs[:, attr_id] == 1, 'int')
                # Mask the similarities between train and test classes, so that similarity for
                # classes having current attribute 'attr_id' NOT ACTIVE will be zeroed.
                masked_similarities = self.class_attribute_similarity[test_cls] * classes_mask
                # We normalize the masked similarity to compute the probability
                prob = softmax(masked_similarities)
                attr_to_train_cls_probs.append(prob)
            P.append(np.array(attr_to_train_cls_probs))
        self._probs = P
        self._idx_active_attr = idx_active_attr
        self._idx_with_label = idx_with_label

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        cls = np.random.randint(len(self._test_attrs))
        return self.get_item_with_class(cls)

    def get_items_with_class(self, Y):
        ENC, CLS = [], []
        for cls in Y:
            enc, y = self.get_item_with_class(cls)
            #AE.append(ae[None, :]); CE.append(ce[None, :]), CLS.append(y)
            ENC.append(enc[None, :]); CLS.append(y)
        return  torch.cat(ENC),  torch.tensor(CLS)


    def get_item_with_class(self, cls):
        attr = self._test_attrs[cls]
        attr_indices, = np.where(attr)
        frnk_enc = [np.zeros_like(E[0]).reshape([self._nb_attributes, -1]) for E in self.encodings]
        for attr_id in attr_indices:
            try:
                # classes_mask = np.asarray(self._train_attrs[:, attr_id] == 1, 'int')
                # masked_similarities = self.class_attribute_similarity[cls] * classes_mask
                # prob1 = softmax(masked_similarities)
                prob = self._probs[cls][attr_id]
                rnd_similar_cls = np.random.choice(np.arange(len(prob)), p=prob)
                example_id = self._idx_with_label[rnd_similar_cls][np.random.randint(len(self._idx_with_label[rnd_similar_cls]))]

            except ValueError as t:
                # In this case there are no  examples with this attribute
                continue
            for i in range(len(self.encodings)):
                frnk_enc[i][attr_id, :] = self.encodings[i][example_id, attr_id, :]

        for i in range(len(self.encodings)):
            frnk_enc[i] = torch.tensor(frnk_enc[i].reshape(-1))
        cls = cls + self._new_class_offset
        return frnk_enc, torch.tensor(cls)



class AsyncInfiniteDataset(Dataset):
    def __init__(self, len, encoder_fn, train_feats, train_labels, train_attrs_bin, test_attrs_bin,
                 ka_dim, kc_dim,
                 new_class_offset=0, device=None, **args):
        """
        :param len: dataset length
        :param encoder_fn: function that take a batch of image features X and return two tensor containing attribute_embedding and
        context_embedding
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
        self._ka_dim = ka_dim
        self._kc_dim = kc_dim
        valid = []

        train_examples_attrs_bin = train_attrs_bin[train_labels]
        for att in range(self._nb_attributes):
            valid.append(np.where(train_examples_attrs_bin[:, att] == 1)[0])
        self._valid = valid

        # Extract attirbute embeddings for training set.
        self._dataset = TensorDataset(torch.tensor(train_feats).float(), torch.tensor(train_labels).long())
        self._attr_encoding = {}
        self._cntx_encoding = {}
        self.device = device

        # self._cntx_encoding = torch.cat(cntx_encoding)
        # self._attr_encoding = attr_encoding.view([attr_encoding.shape[0], self._nb_attributes, -1])

        # Find valid examples for each attribute.

    # def compute_encodings(self, i):
    #     x, y = self._dataset[i]
    #     x = x.to(self.device)
    #     ae, ce = self._encoder(x)
    #     ae = ae.detach().cpu().view([self._nb_attributes, -1])
    #     ce =  ce.detach().cpu()
    #     self._attr_encoding[i] = ae
    #     self._cntx_encoding[i] = ce
    #     return ae, ce
    #
    # def encoding(self, i):
    #     try:
    #         return self._attr_encoding[i], self._cntx_encoding[i]
    #     except KeyError:
    #         return self.compute_encodings(i)
    #
    # def attr_encoding(self, i):
    #     return self.encoding(i)[0]
    #
    # def cntx_encoding(self, i):
    #     return self.encoding(i)[1]
    def encode(self, idx):
        x, y = self._dataset[idx]
        x = x.to(self.device)
        ae, ce = self._encoder(x)
        ae = ae.detach().view([len(x), self._nb_attributes, -1])
        ce =  ce.detach()
        return ae, ce

    def clear_cache(self):
        self._attr_encoding.clear()
        self._cntx_encoding.clear()


    def __len__(self):
        return self._len

    # def __getitem__(self, i):
    #     cls = np.random.randint(len(self._test_attrs))
    #     return self.get_item(cls)

    def __getitem__(self, i):
        return self.encoding(i)

    #
    # def get_items_with_class(self, Y):
    #     AE, CE, CLS = [], [], []
    #     for cls in Y:
    #         ae, ce, y = self.get_item_with_class(cls)
    #         AE.append(ae[None, :]);
    #         CE.append(ce[None, :]), CLS.append(y)
    #     return torch.cat(AE), torch.cat(CE), torch.tensor(CLS)
    #

    def get_items_with_class(self, Y):
        attribute_vectors = self._test_attrs[Y]
        #attribute_vectors = self._test_attrs[torch.unique(Y)]

        attr_indices = []
        for attr_vec in attribute_vectors:
            attr_indices.append(np.where(attr_vec)[0])

        #attr_indices = (torch.tensor(attr_vec)==1).nonzero()

        random_idx = np.random.randint(len(self._dataset), size=128)
        _, random_cntx_enc = self.encode(random_idx)

        data_indices = []
        for i, attr_idx in enumerate(attr_indices):
            data_idx = []
            for a_id in attr_idx:
                try:
                    data_id = self._valid[a_id][np.random.randint(len(self._valid[a_id]))]
                    data_idx.append(data_id)
                except ValueError as t:
                    # In this case there are no  examples with this attribute
                    continue
            data_indices.append(torch.tensor(data_idx))
        selected_data_indices = torch.cat(data_indices, dim=0).unique()
        selected_attr_emb, selected_cntx_emb = self.encode(selected_data_indices)
        data_id_to_selected_attr_id = {int(index):i for i, index in enumerate(selected_data_indices)}

        frankenstein_attr_enc = torch.zeros([len(Y), self._nb_attributes, self._ka_dim]).float()
        for i, (attr_idx, data_idx) in enumerate(zip(attr_indices, data_indices)):
            for a_id, data_id in zip(attr_idx, data_idx):
                frankenstein_attr_enc[i, a_id, :] = selected_attr_emb[data_id_to_selected_attr_id[int(data_id)], :][a_id]

        frankenstein_attr_enc = frankenstein_attr_enc.reshape(len(Y), -1)
        Y = Y + self._new_class_offset
        return torch.tensor(frankenstein_attr_enc), random_cntx_enc, torch.tensor(Y)

    def get_item_with_class(self, cls):
        attr = self._test_attrs[cls]
        attr_indices = np.where(attr)[0]

        random_idx = np.random.randint(len(self._dataset))
        random_cntx_enc = self.cntx_encoding(random_idx)
        frankenstein_attr_enc = np.zeros([self._nb_attributes, self._ka_dim], dtype='float32')
        for attr_idx in attr_indices:
            try:
                emb = self._valid[attr_idx][np.random.randint(len(self._valid[attr_idx]))]
            except ValueError as t:
                # In this case there are no  examples with this attribute
                # t.with_traceback()
                # raise t
                continue
            frankenstein_attr_enc[attr_idx, :] = self.attr_encoding(emb)[attr_idx, :]

        frankenstein_attr_enc = frankenstein_attr_enc.reshape(-1)
        cls = cls + self._new_class_offset
        return torch.tensor(frankenstein_attr_enc), random_cntx_enc, torch.tensor(cls)


def FrankensteinDataset(per_class_samples, encoder_fn, train_feats, train_labels, train_attrs_bin, test_attrs_bin,
                        new_class_offset=0, device=None):
        nb_attributes = train_attrs_bin.shape[1]
        # Extract attirbute embeddings for training set.
        ds = TensorDataset(torch.tensor(train_feats).float(), torch.tensor(train_labels).long())
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=6, pin_memory=False, drop_last=False)
        attr_encoding = []
        cntx_encoding = []
        for X in dl:
            X = X[0].to(device)
            ae, ce = encoder_fn(X)
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

