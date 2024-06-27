# setup environment
import os
smoke_test = ('CI' in os.environ)  # for continuous integration tests

import scanpy as sc
import torch
import torch.nn as nn
import math
import numpy as np
import anndata as ad
import torch
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset,DataLoader,Sampler
from scipy.sparse import csr_matrix

def data_load(anndata, n_top_genes, batch_key = 'batch_2', condition_key = 'batch', groundtruth_key = None):
    ######### 该代码用于预处理scanpy的原始数据，将单细胞测序数据进行清洗和质控，并且normalized它们 ##########
    datasets = anndata.copy()
    if isinstance(datasets.X, csr_matrix):
        # 如果是稀疏矩阵，使用toarray()方法转换为NumPy数组，然后进行四舍五入
        datasets.X = np.round(datasets.X.toarray()).astype(int)
    else:
        # 如果是NumPy数组，直接进行四舍五入
        datasets.X = np.round(datasets.X).astype(int)

    datasets.obs['batch'] = datasets.obs[condition_key].astype('category')
    datasets.obs['batch_2'] = datasets.obs[batch_key].astype('category')
    if groundtruth_key is not None:
        datasets.obs['groundtruth'] = datasets.obs[groundtruth_key].astype('category')
    datasets.layers['counts'] = datasets.X.copy()
    # 数据筛选
    sc.pp.filter_cells(datasets, min_genes=200)
    sc.pp.filter_genes(datasets, min_cells=3)
    raw = datasets.copy()
    # 数据标准化
    sc.pp.normalize_total(datasets, target_sum=1e4)
    # 进行log1p转换
    sc.pp.log1p(datasets)
    datasets.layers['lognorm'] = datasets.X.copy()
    # 识别高变基因
    sc.pp.highly_variable_genes(datasets, batch_key= None, n_top_genes=n_top_genes)
    # 将高变基因存在var中
    datasets.raw = datasets
    # 保留高变基因
    datasets = datasets[:, datasets.var.highly_variable]
    # scale数据
    sc.pp.scale(datasets, max_value=10)

    # batch = np.unique(datasets.obs[batch_key])
    # list_batch = []
    # for x in batch:
    #     adata_iter = datasets[datasets.obs[batch_key] == x]
    #     sc.pp.normalize_total(adata_iter, target_sum=1e5)
    #     list_batch.append(adata_iter)

    # datasets = ad.concat(list_batch)
    # datasets = sc.pp.log1p(datasets, copy = True)
    # datasets.layers['lognorm'] = datasets.X.copy()
    # # 识别高变基因
    # sc.pp.highly_variable_genes(datasets, n_top_genes=n_top_genes, batch_key='batch')
    # # 将高变基因存在var中
    # datasets.raw = datasets
    # # 保留高变基因
    # datasets = datasets[:, datasets.var.highly_variable]

    var_names = datasets.var_names
    index_names = datasets.obs_names
    
    return datasets,raw,var_names,index_names

# class BatchDataLoader(object):
#     """
#     This custom DataLoader serves mini-batches that are either fully-observed (i.e. labeled)
#     or partially-observed (i.e. unlabeled) but never mixed.
#     """

#     def __init__(self, data_x, data_y, batch_size, num_classes, missing_label=-1):
#         super().__init__()
#         self.data_x = data_x
#         self.data_y = data_y
#         self.batch_size = batch_size
#         self.num_classes = num_classes

#         self.unlabeled = torch.where(data_y == missing_label)[0]
#         self.num_unlabeled = self.unlabeled.size(0)
#         self.num_unlabeled_batches = math.ceil(self.num_unlabeled / self.batch_size)

#         self.labeled = torch.where(data_y != missing_label)[0]
#         self.num_labeled = self.labeled.size(0)
#         self.num_labeled_batches = math.ceil(self.num_labeled / self.batch_size)

#         self.classed = [torch.where(data_y == c)[0] for c in range(num_classes)]
#         self.num_classed = [c.size(0) for c in self.classed]
#         self.num_class_batches = [math.ceil(indices / self.batch_size) for indices in self.num_classed]

#         assert self.data_x.size(0) == self.data_y.size(0)
#         assert len(self) > 0

#     @property
#     def size(self):
#         return self.data_x.size(0)

#     def __len__(self):
#         return self.num_unlabeled_batches + self.num_labeled_batches

#     def _sample_batch_indices(self):
#         batch_order = torch.randperm(len(self)).tolist()
#         unlabeled_idx = self.unlabeled[torch.randperm(self.num_unlabeled)]
#         labeled_idx = self.labeled[torch.randperm(self.num_labeled)]
#         class_idx = [self.classed[c][torch.randperm(self.num_classed[c])] for c in range(self.num_classes)]

#         # print(class_idx[0].shape,class_idx[1].shape)

#         slices = []

#         for i in range(self.num_unlabeled_batches):
#             _slice = unlabeled_idx[i * self.batch_size : (i + 1) * self.batch_size]
#             slices.append((_slice, False))

#         # for i in range(self.num_labeled_batches):
#         #     _slice = labeled_idx[i * self.batch_size : (i + 1) * self.batch_size]
#         #     slices.append((_slice, True))

#         for c in range(self.num_classes):
#             for i in range(self.num_class_batches[c]):
#                 _slice = class_idx[c][i * self.batch_size : (i + 1) * self.batch_size]
#                 slices.append((_slice, True))

#         return slices, batch_order

#     def __iter__(self):
#         slices, batch_order = self._sample_batch_indices()

#         for i in range(len(batch_order)):
#             _slice = slices[batch_order[i]]
#             if _slice[1]:
#                 # labeled
#                 yield self.data_x[_slice[0]], nn.functional.one_hot(
#                     self.data_y[_slice[0]], num_classes=self.num_classes
#                 )
#             else:
#                 # unlabeled
#                 yield self.data_x[_slice[0]], None

# class BatchDataLoader2(object):
#     """
#     This custom DataLoader serves mini-batches that are either fully-observed (i.e. labeled)
#     or partially-observed (i.e. unlabeled) but never mixed.
#     """

#     def __init__(self, data_x, data_y, batch_size, num_classes=2, missing_label=-1):
#         super().__init__()
#         self.data_x = data_x
#         self.data_y = data_y
#         self.batch_size = batch_size
#         self.num_classes = num_classes

#         self.unlabeled = torch.where(data_y == missing_label)[0]
#         self.num_unlabeled = self.unlabeled.size(0)
#         self.num_unlabeled_batches = math.ceil(self.num_unlabeled / self.batch_size)

#         self.labeled = torch.where(data_y != missing_label)[0]
#         self.num_labeled = self.labeled.size(0)
#         self.num_labeled_batches = math.ceil(self.num_labeled / self.batch_size)

#         assert self.data_x.size(0) == self.data_y.size(0)
#         assert len(self) > 0

#     @property
#     def size(self):
#         return self.data_x.size(0)

#     def __len__(self):
#         return self.num_unlabeled_batches + self.num_labeled_batches

#     def _sample_batch_indices(self):
#         batch_order = torch.randperm(len(self)).tolist()
#         unlabeled_idx = self.unlabeled[torch.randperm(self.num_unlabeled)]
#         labeled_idx = self.labeled[torch.randperm(self.num_labeled)]

#         slices = []

#         for i in range(self.num_unlabeled_batches):
#             _slice = unlabeled_idx[i * self.batch_size : (i + 1) * self.batch_size]
#             slices.append((_slice, False))

#         for i in range(self.num_labeled_batches):
#             _slice = labeled_idx[i * self.batch_size : (i + 1) * self.batch_size]
#             slices.append((_slice, True))

#         return slices, batch_order

#     def __iter__(self):
#         slices, batch_order = self._sample_batch_indices()

#         for i in range(len(batch_order)):
#             _slice = slices[batch_order[i]]
#             if _slice[1]:
#                 # labeled
#                 yield self.data_x[_slice[0]], nn.functional.one_hot(
#                     self.data_y[_slice[0]], num_classes=self.num_classes
#                 )
#             else:
#                 # unlabeled
#                 yield self.data_x[_slice[0]], None

def transform_numpy(data, cuda):
    if cuda==True:
        data = torch.Tensor(data).cuda()
    else:
        data = torch.Tensor(data)
    return data

def shuffer(adata):
    idx = np.random.permutation(len(adata.obs_names))
    shuffled_obs_names = adata.obs_names[idx]
    adata = adata[shuffled_obs_names]
    return adata

# class BatchSampler(Sampler):
#     def __init__(self, adata, batch_size, label_encoder = LabelEncoder()):
#         self.labels = torch.Tensor(label_encoder.fit_transform(adata.obs['batch'])).cuda()
#         self.batch_size = batch_size
#         self.num_labels = torch.unique(self.labels)

#         self.classed = [torch.where(self.labels == c)[0] for c in range(len(self.num_labels))]
#         self.counts_labels = [c.size(0) for c in self.classed]
#         self.num_class_batches = [math.ceil(indices / self.batch_size) for indices in self.counts_labels]

#     def __iter__(self):
#         iter_labels = self.labels # 可迭代的
#         indices = {} # 用于储存不同的labels对应的index
#         for c in self.num_labels.tolist():
#             indices[c] = [i for i, label in enumerate(self.labels) if label == c]
#         while len(iter_labels) > 0:
#             batch_labels = iter_labels[0].item()
#             select_indices = indices[batch_labels]
#             indices_iter = [i for i, label in enumerate(iter_labels) if label == batch_labels]

#             if len(select_indices) >= self.batch_size:
#                 yield select_indices[:self.batch_size]
#                 indices[batch_labels] = list(set(indices[batch_labels]) - set(select_indices[:self.batch_size]))
#                 iter_labels = torch.Tensor([label for i, label in enumerate(iter_labels) if i not in indices_iter[:self.batch_size]])
#             else:
#                 yield select_indices
#                 indices[batch_labels] = []
#                 iter_labels = torch.Tensor([label for i, label in enumerate(iter_labels) if i not in indices_iter])
                
#     def __len__(self):
#         return np.sum(self.num_class_batches)

class BatchSampler(Sampler):
    def __init__(self, adata, batch_size, cuda, label_encoder = LabelEncoder()):
        self.cuda = cuda
        if cuda:
            self.labels = torch.tensor(label_encoder.fit_transform(adata.obs['batch'])).cuda()
        else:
            self.labels = torch.tensor(label_encoder.fit_transform(adata.obs['batch'])).cpu()
        self.batch_size = batch_size
        self.num_labels = torch.unique(self.labels)
        self.indices = {}  # Precompute indices

        for c in self.num_labels.tolist():
            self.indices[c] = torch.where(self.labels == c)[0]

        self.num_class_batches = [math.ceil(len(indices) / self.batch_size) for indices in self.indices.values()]
        self.iter_labels = [torch.ones(n)*i for i,n in enumerate(self.num_class_batches)]
        self.iter_labels = torch.cat(self.iter_labels).int().tolist()

    def __iter__(self):
        iter_indices = self.indices.copy()
        for batch_labels in self.iter_labels:
            select_indices = iter_indices[batch_labels]
            if len(select_indices) >= self.batch_size:
                yield select_indices[:self.batch_size]
                iter_indices[batch_labels] = select_indices[self.batch_size:]
            else:
                yield select_indices
                iter_indices[batch_labels] = torch.tensor([], dtype=torch.long)  # Empty tensor

    def __len__(self):
        return np.sum(self.num_class_batches)


class scDataset(Dataset):
    def __init__(self, adata, cuda, transform=None):
        super().__init__()
        self.batch_label = pd.get_dummies(adata.obs['batch_2']).values
        self.condition_label = pd.get_dummies(adata.obs['batch']).values
        self.expr = adata.X # 注意需不需要toarray()
        self.transform = transform
        self.cuda = cuda

    def __len__(self):
        return len(self.batch_label)
    
    def __getitem__(self, index):
        expr = self.expr[index,:]
        batch_label = self.batch_label[index,:]
        condition_label = self.condition_label[index,:]
        ifcuda = self.cuda
        if self.transform:
            expr = transform_numpy(expr,ifcuda)
            batch_label = transform_numpy(batch_label,ifcuda)
            condition_label = transform_numpy(condition_label,ifcuda)
        return expr,condition_label,batch_label