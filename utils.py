# setup environment
import os
smoke_test = ('CI' in os.environ)  # for continuous integration tests

import torch
import scanpy as sc
# from scib import me
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

def process_batch_X(datasets):
    ########### 这个函数用于提取基因表达阵x 和 条件信息batch ########### 
    batch = datasets.obs['batch']
    batch_labels = batch.unique()
    try:
        datasets.X = datasets.X.todense() # 所有数据
        print('稀疏矩阵转为密集型矩阵！')
    except:
        pass
    x = torch.tensor(datasets.X)
    x_list = [torch.tensor(datasets[datasets.obs['batch'] == label].X) for label in batch_labels] # 各个条件的数据

    datasets.obs['one_hot'] = -1 # 创建batch组的one-hot-vector
    for i, label in enumerate(batch_labels):
        datasets.obs.loc[datasets.obs['batch'] == label, 'one_hot'] = i
    
    one_hot = torch.tensor(datasets.obs['one_hot'].tolist()) # 将其存成tensor格式
    x = x.to(torch.float32)
    x_list = [x.to(torch.float32) for x in x_list]

    return x, x_list, batch, batch_labels, one_hot

def hyper_param(x, x_list, batch_labels):
    ########### 这个函数用于计算模型需要的一些参数，包括基因数量、条件数量以及各条件下基因表达的均值和方差 ########### 
    num_genes = x.shape[1]
    num_batch = len(batch_labels)

    x_list_mean = [x.mean(axis=0) for x in x_list]
    x_list_scale = [x.std(axis=0) for x in x_list]

    return num_genes, num_batch, x_list_mean, x_list_scale

# def Find_ari(datasets):
#     ########### 这个函数用于寻找模型的最优聚类 ########### 
#     ari_max = -1
#     index_i = 0
#     for i in range(0,30):
#         sc.tl.louvain(datasets,resolution=i*0.01,key_added='seurat_clusters')
#         ari = me.ari(datasets,'groundtruth','seurat_clusters')
#         if ari>ari_max:
#             ari_max = ari
#             index_i = i
#     return ari_max,index_i


def construct_anndata(data,columns_names,index_names,raw):
    ########### 这个函数构建新的anndata用来进行下游分析 ########### 
    DATA = pd.DataFrame(data=data, columns=columns_names,index=index_names)
    datasets = sc.AnnData(DATA)
    datasets.obs['sample_id'] = DATA.index  # 添加样本 ID 列
    datasets.var['gene_name'] = DATA.columns  # 添加基因名列

    datasets.raw = datasets
    datasets.obs = raw.obs

    datasets.layers['lognorm'] = datasets.X.copy()
    # scale数据这是用于PCA的！！
    sc.pp.scale(datasets, max_value=8)
    datasets.layers['scale'] = datasets.X.copy()
    datasets.layers['counts'] = np.expm1(datasets.layers['scale']-1).astype(int) # 反向操作
    # sc.tl.pca(datasets, svd_solver='arpack')
    # sc.pp.neighbors(datasets, n_neighbors=50, n_pcs=25)
    # sc.tl.umap(datasets)
    
    return datasets

def create_hyper(datasets,var_names,index_names,batch_size=100):
    x, x_list, batch, batch_labels, one_hot = process_batch_X(datasets)
    num_genes, num_batch, x_list_mean, x_list_scale = hyper_param(x, x_list, batch_labels)
    num_genes = x.shape[1]
    batch_size = batch_size
    hyper = {'num_genes': num_genes,
    'num_batch':num_batch,
    'batch_size':batch_size,
    'batch_labels':batch_labels,
    'one_hot':one_hot,
    'x':x,
    'x_list':x_list,
    'batch':batch,
    'var_names':var_names,
    'index_names':index_names
    }
    return hyper


def fit_knn(mat_train, mat_holdout, n_neighbors, algorithm = 'kd_tree'):
    
    # fit knn using mat_train
    # return nn indices and distances in train set for holdout set
    knn = NearestNeighbors(n_neighbors = n_neighbors, algorithm = algorithm).fit(mat_train)
    distances, indices = knn.kneighbors(mat_holdout)
    indices = indices[:,1:]
    distances = distances[:,1:]

    return indices, distances



def calc_knn_prop(knn_indices, labels_train, label_categories):

    # knn_indices: shape = (n_holdout_samples, (knn-1)), np.array
    # labels_train: shape = (n_train_samples, ), pd.object
    # label_categories: shape = (n_label_categories, ), np.array
    n = knn_indices.shape[0]
    n_category = label_categories.shape[0]
    nn_prop = np.zeros(shape = (n, n_category))

    for i in range(n):
        knn_labels = labels_train[knn_indices[i,]]
        for k in range(n_category):
            nn_prop[i, k] = sum(knn_labels == label_categories[k]) 

    nn_prop = nn_prop / knn_indices.shape[1]
    return nn_prop


def calc_oobNN(adata_orig, batch_key, condition_key, n_neighbors=15, holdout=True):
    np.random.seed(123)
    
    list_holdout = []
    for holdout_idx in np.unique(adata_orig.obs[batch_key]):
        if holdout == True:
            adata_train = adata_orig[~adata_orig.obs[batch_key].isin([holdout_idx])]
            adata_holdout = adata_orig[adata_orig.obs[batch_key].isin([holdout_idx])]
        else:
            adata_train = adata_orig
            adata_holdout = adata_orig[adata_orig.obs[batch_key].isin([holdout_idx])]
        num_cells = adata_train.obs[condition_key].value_counts().min()

    
        a_list = []
        for x in np.unique(adata_train.obs[condition_key]):
            a1 = adata_train[adata_train.obs[condition_key].isin([x])]
            random_indices = np.random.choice(a1.shape[0], size=num_cells, replace = False)
            a1 = a1[random_indices,:]
            a_list.append(a1)
 
        adata_train = ad.concat(a_list)
        adata = ad.concat([adata_train, adata_holdout])
    
        mat = sc.tl.pca(adata.X, n_comps = 20)

        mat_train = mat[:adata_train.obs.shape[0],]
        mat_holdout = mat[adata_train.obs.shape[0]:,]

    
        # fit knn
        indices, distances = fit_knn(mat_train=mat_train, mat_holdout=mat_holdout, n_neighbors=n_neighbors, algorithm = 'kd_tree')

        # compute proprotion
        labels_train = adata_train.obs[condition_key].astype('object')
        label_categories = np.unique(labels_train)
        result = calc_knn_prop(indices, labels_train, label_categories)
        knn_df = pd.DataFrame(data=result, 
                              index =  adata_holdout.obs_names,
                              columns = label_categories)
        adata_holdout.obsm['knn_prop'] = knn_df
        list_holdout.append(adata_holdout)

    res = ad.concat(list_holdout)
    return res


def calc_oobNN2(adata_orig, batch_key, n_neighbors=15):
    np.random.seed(123)
    
    list_holdout = []
    for holdout_idx in np.unique(adata_orig.obs[batch_key]):

        adata_train = adata_orig[~adata_orig.obs[batch_key].isin([holdout_idx])]
        adata_train.obs['cobatch'] = 'Rest'
        adata_holdout = adata_orig[adata_orig.obs[batch_key].isin([holdout_idx])]
        adata_holdout.obs['cobatch'] = 'Self'

        num_cells = min(adata_train.obs.shape[0],adata_holdout.obs.shape[0])

        a_list = []
        for a1 in [adata_train,adata_holdout]:
            random_indices = np.random.choice(a1.shape[0], size=num_cells, replace = False)
            a1 = a1[random_indices,:]
            a_list.append(a1)

        adata_train = ad.concat(a_list)
        adata = ad.concat([adata_train, adata_holdout])
    
        mat = sc.tl.pca(adata.X, n_comps = 20)

        mat_train = mat[:adata_train.obs.shape[0],]
        mat_holdout = mat[adata_train.obs.shape[0]:,]

    
        # fit knn
        indices, distances = fit_knn(mat_train=mat_train, mat_holdout=mat_holdout, n_neighbors=n_neighbors, algorithm = 'kd_tree')

        # compute proprotion
        labels_train = adata_train.obs['cobatch'].astype('object')
        label_categories = np.unique(labels_train)
        result = calc_knn_prop(indices, labels_train, label_categories)
        knn_df = pd.DataFrame(data = result, 
                              index =  adata_holdout.obs_names,
                              columns = label_categories)
        adata_holdout.obsm['knn_prop'] = knn_df
        list_holdout.append(adata_holdout)
    
    res = ad.concat(list_holdout)
    return res