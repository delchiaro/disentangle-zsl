import torch
import numpy as np
from torch import nn
from torchmeta.transforms import ClassSplitter, Categorical
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from MetaTensorDataset import TensorMetaDataset
from disentangle.layers import L2Norm, BlockLinear
from utils import NP

from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Callable, Union, List, Tuple, Union, _VariadicGenericAlias, TypeVar
import data

from disentangle.loss_box import LossBox
from utils import init
from sklearn import metrics

import torch.nn.functional as NN
from torch.nn import functional as F
from disentangle.utils import set_grad
import torch.distributions as tdist


#%% INIT
def init_exp(gpu, seed, use_valid=False, download_data=False, preprocess_data=False, dataset='AWA2', attr_in_01=True):
    device = init(gpu_index=gpu, seed=seed)
    print('\n\n\n' + ('*' * 80) + f'\n   Starting new exp with seed {seed} on device {device}\n' + ('*' * 80) + '\n')
    if download_data:
        data.download_data()
    if preprocess_data:
        data.preprocess_dataset(dataset)
    train, val, test_unseen, test_seen = data.get_dataset(dataset, use_valid=use_valid, gzsl=False, mean_sub=False, std_norm=False,
                                                          l2_norm=False)
    if attr_in_01:
        train, val, test_unseen, test_seen = data.normalize_dataset(train, val, test_unseen, test_seen, keys=('class_attr', 'attr'),
                                                                    feats_range=(0, 1))

    return device, train, test_unseen, test_seen, val

def torch_where_idx(condition):
    return condition.byte().nonzero()[:,0]

#%% NETWORKS
from disentangle.model import Model

class CosineLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__(in_features, out_features, bias=False)

    def forward(self, x):
        x_norm = x.norm(p=2, dim=1)[:, None]
        w_norm = self.weight.norm(p=2, dim=1)[:, None]
        #self.weight.data /= w_norm
        return F.linear(x/x_norm, self.weight/w_norm)
        #return F.linear(x, self.weight)/x_norm.matmul(w_norm.transpose(0, 1))


class Encoder(Model):
    def __init__(self, nb_attr, in_feats=2048, hidden_dim=2048, latent_attr_dim=64):
        super().__init__()
        self.nb_attr = nb_attr
        self.in_feats = in_feats
        self.hidden_dim = 2048
        self.latent_attr_dim = 2048
        self.net = nn.Sequential(nn.Linear(in_feats, hidden_dim), nn.ReLU(),
                                 #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, latent_attr_dim * nb_attr))
    def forward(self, x):
        return self.net(x)

class AttrDecoder(Model):
    def __init__(self, nb_attr, hidden_dim=2048, latent_attr_dim=64):
        super().__init__()
        self.nb_attr = nb_attr
        self.hidden_dim = 2048
        self.latent_attr_dim = 2048
        self.net = nn.Sequential(#nn.Linear(latent_attr_dim * nb_attr, hidden_dim), nn.ReLU(),
                                 #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, nb_attr))
    def forward(self, x):
        return self.net(x)



#%% TEST

#%% MAIN
from torch import optim
def to(tensors, *args, **kwargs):
    return (t.to(*args, **kwargs) for t in tensors)

T = TypeVar('T')
ListOrTuple = Union[List[T], Tuple[T]]

class TensorTaskDataset:
    def __init__(self, per_class_tensors: ListOrTuple[ListOrTuple[torch.Tensor]], classes: List[int]):
        self._per_class_tensors = per_class_tensors
        self._classes = classes
        self._nb_classes = len(classes)
        self._global2local = {cls: i for i, cls in enumerate(self._classes)}
        self._local2global = {i: cls for i, cls in enumerate(self._classes)}

    def local2global(self, local_label):
        return self._local2global[local_label]

    def locals2globals(self, local_labels):
        if isinstance(local_labels, torch.Tensor):
           local_labels = local_labels.cpu().numpy()
        local_labels = list(local_labels)
        return [self._local2global[ll] for ll in local_labels]

    def global2local(self, global_label):
        return self._global2local[global_label]

    def globals2locals(self, global_labels):
        if isinstance(global_labels, torch.Tensor):
           local_labels = global_labels.cpu().numpy()
        local_labels = list(global_labels)
        return [self._global2local[gl] for gl in global_labels]

    def get_data_loader(self, batch_size=32, shuffle=False, drop_last=True, max_batches=None):
        def rebalanced_data_loader(batch_size=32, shuffle=False, drop_last=True, max_batches=None):
            per_class_examples = []
            per_class_first_idx = []
            permuted_cls_tensors = []
            next_first = 0
            for cls_tensors in self._per_class_tensors:
                nb_cls_examples = cls_tensors[0].shape[0]
                per_class_examples.append(nb_cls_examples)
                per_class_first_idx.append(next_first)
                next_first += nb_cls_examples
                if shuffle:
                    perm = torch.randperm(nb_cls_examples)
                    permuted_cls_tensors.append([t[perm] for t in cls_tensors])
            nb_examples = next_first
            catted_tensors = [torch.cat([t[i] for t in permuted_cls_tensors]) for i in range(len(permuted_cls_tensors[0]))]
            if max_batches is None:
                max_batches = len(catted_tensors[0])+1

            end = False
            local_classes = np.arange(len(self._classes))
            batch = 0
            while (not end):

                labels = np.random.choice(local_classes, batch_size)
                per_class_nb_examples = np.bincount(labels, minlength=self._nb_classes)
                batch_example_idx = []
                per_class_idx_counter = [0 for _ in self._per_class_tensors]

                for c, nb_examples in enumerate(per_class_nb_examples):
                    first_elem = per_class_idx_counter[c]
                    last_elem = first_elem + nb_examples
                    if first_elem > per_class_examples[c] or (drop_last and last_elem > per_class_examples[c]):
                        end = True
                        break
                    else:
                        last_elem = min(last_elem, per_class_examples[c])
                    batch_example_idx.append(torch.arange(per_class_first_idx[c] + first_elem, per_class_first_idx[c] + last_elem, dtype=torch.long))
                batch_example_idx = torch.cat(batch_example_idx)
                if not end:
                    yield [t[batch_example_idx] for t in catted_tensors] + [torch.tensor(labels)]
                batch += 1
                if batch == max_batches:
                    end = True

        return rebalanced_data_loader(batch_size, shuffle, drop_last, max_batches)


class TensorMetaDataset:
    def __init__(self, tensors: Union[List[torch.Tensor], Tuple[torch.Tensor]], targets: torch.Tensor, device=None):
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        elif not isinstance(targets, np.ndarray):
            targets = np.array(targets)

        self._device = device
        self._targets = targets
        self._classes = sorted(set(targets))
        self._num_classes = len(self._classes)
        self._per_class_tensors = {}
        for cls in self._classes:
            idx = np.argwhere(self._targets == cls)[:, 0]
            self._per_class_tensors[cls] = tuple(t[idx] for t in tensors)

    def sample_task(self, classes) -> TensorTaskDataset:
        return TensorTaskDataset([self._per_class_tensors[cls] for cls in classes], classes)

    def sample_new_task(self, meta_train_classes=20, meta_test_classes=0, meta_valid_classes=0) -> List[TensorTaskDataset]:
        meta_train_classes = max(meta_train_classes, 0)
        meta_test_classes = max(meta_test_classes, 0)
        meta_valid_classes = max(meta_valid_classes, 0)
        assert self._num_classes >= meta_train_classes + meta_test_classes + meta_valid_classes
        cls_perm = np.random.permutation(self._classes)
        tasks = []
        if meta_train_classes > 0:
            tasks.append(self.sample_task(cls_perm[:meta_train_classes]))  # train task
        if meta_test_classes > 0:
            tasks.append(self.sample_task(cls_perm[meta_train_classes:meta_test_classes]))  # test task
        if meta_valid_classes > 0:
            tasks.append(self.sample_task(cls_perm[meta_train_classes + meta_test_classes:meta_valid_classes]))  # valid task
        return tasks


if __name__ == '__main__':
    NB_TASKS = 10000
    TASK_BATCHES = 5
    ATTR_IN_01=True
    #device, train, test_unseen, test_seen, val = init_exp(gpu=0,seed=None, dataset='AWA2', attr_in_01=ATTR_IN_01)
    device, train, test_unseen, test_seen, val = init_exp(gpu=0, seed=42, dataset='AWA2', attr_in_01=ATTR_IN_01)
    test_period = 2
    lr = .001
    wd = 0
    momentum=.7
    latent_attr_dim=4

    # MANAGE DATA
    nb_train_examples, feats_dim = train['feats'].shape
    nb_train_classes, nb_attributes = train['class_attr'].shape
    nb_test_classes, _ = test_unseen['class_attr'].shape
    A_mask = torch.tensor(train['class_attr_bin']).float()
    A = torch.tensor(train['class_attr']).float()
    train_attribute_masks = torch.tensor(train['attr_bin']).float()
    train_attributes = torch.tensor(train['attr']).float()
    train_features = torch.tensor(train['feats']).float()
    train_labels = torch.tensor(train['labels']).long()
    #trainset = TensorDataset(train_features, train_labels, train_attribute_masks, train_attributes)
    #train_loader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)
    #dataset = TensorDataset(train_features, train_attributes, train_labels)

    meta_tensor_dataset = TensorMetaDataset((train_features, train_attributes), train_labels)


    for task in range(NB_TASKS):
        train_task, test_task = meta_tensor_dataset.sample_new_task(30, 10)
        train_loader = train_task.get_data_loader(128, shuffle=True, max_batches=TASK_BATCHES)
        for b, (X, attr, Y) in enumerate(train_loader):
            print(f"Batch {b+1}")
            #print('Train inputs shape: {0}'.format(X.shape))  # (16, 25, 1, 28, 28)
            #print('Train targets shape: {0}'.format(Y.shape))  # (16, 25)
            #print(f'Local-Labels: {Y}')
            #print(f'Global-Labels: {train_task.locals2globals(Y)}')

        E = Encoder(nb_attributes, feats_dim, hidden_dim=2048, latent_attr_dim=latent_attr_dim).to(device)
        #AD = CosineLinear(nb_attributes*latent_attr_dim, nb_attributes).to(device)
        # AD = nn.Linear(nb_attributes*latent_attr_dim, nb_attributes).to(device)
        AD = BlockLinear(latent_attr_dim, 1, nb_attributes).to(device)
        C = CosineLinear(nb_attributes*latent_attr_dim, nb_train_classes).to(device)
        #C = nn.Linear(nb_attributes*latent_attr_dim, nb_train_classes).to(device)


        #opt = optim.SGD(list(E.parameters()) + list(C.parameters()), lr=lr, weight_decay=wd, momentum=momentum)
        opt = optim.Adam(list(E.parameters()) + list(C.parameters()), lr=lr, weight_decay=wd)


        def get_norm_W():
            return NP(C.weight / C.weight.norm(p=2, dim=1)[:, None])

        def get_avg_norm_feats():
            tloader = DataLoader(TensorDataset(train_features), batch_size=128, shuffle=False, drop_last=False)
            Z = []
            Y = train_labels.numpy()
            for x in tloader:
                Z.append(E(x[0].to(device)).detach().cpu())
            Z = torch.cat(Z)
            Z = Z/Z.norm(p=2, dim=1, keepdim=True)
            Z = Z.numpy()
            Z_perclass = []
            for y in set(Y):
                idx = np.argwhere(Y==y)[:,0]
                Z_perclass.append(np.mean(Z[idx], axis=0))
            return np.array(Z_perclass)

        for ep in range(nb_epochs):
            C.train(); E.train()
            L = LossBox()

            y_preds = []
            y_trues = []
            a_trues = []
            a_preds = []
            for i, (x, y, a_mask, a) in enumerate(train_loader):
                x, y, a_mask, a = to((x, y, a_mask, a), device)
                opt.zero_grad()
                z = E(x)
                logits = C(z)
                a1 = AD(z)
                sm =  torch.softmax(logits, dim=1)

                cce_loss = F.nll_loss(sm, y)  # seems to work better when using cosine
                #cce_loss = F.cross_entropy(logits, y)

                attr_loss = F.mse_loss(a1, a)#, reduction='none').mean(dim=0).sum()
                #attr_loss = F.mse_loss(a1, a)#, reduction='none').mean(dim=0).sum()

                (1 * cce_loss + 1*attr_loss).backward()
                opt.step()
                y_preds.append(logits.argmax(dim=1).detach().cpu())
                y_trues.append(y.detach().cpu())
                a_preds.append((a1>0.5).float().detach().cpu())
                a_trues.append(a_mask.detach().cpu())

                L.append('cce', cce_loss.cpu().detach())
                L.append('attr', attr_loss.cpu().detach())

            y_preds = torch.cat(y_preds)
            y_trues = torch.cat(y_trues)
            a_preds = torch.cat(a_preds)
            a_trues = torch.cat(a_trues)
            acc = (y_preds == y_trues).float().mean()
            attr_acc = (a_preds == a_trues).float().mean()

            print(f'====> Epoch: {ep+1}   -  train-acc={acc}  train-attr-acc-={attr_acc}')
            print(L)

            if (ep+1)%test_period == 0:
                exemplars = torch.tensor(get_avg_norm_feats())
                W = torch.tensor(get_norm_W())
                dist = torch.norm(exemplars-W, p=2, dim=1)
                print(f"Per class l2-distance (weight vs per-class-mean-embedding): {dist.mean()}")
                print(dist)

                # Test on unseen-test-set:
                # TODO: generate weight for classifier of unseen-classes
                # TODO: test on test-set
                pass

#%% TRASHCAN



