# iperturb

iPerutrb使用变分自动编码器实现了人群规模单细胞数据的多条件整合。软胶包的信息参考pipy：https://pypi.org/project/iperturb/；样例信息可以参考:

## 安装

使用 pip 安装：

```bash
pip install iPerturb
```

## 使用

安装完成后，可以在Python代码中使用 iPerturb 包中的函数和类。

例如：

```python
import iPerturb.iperturb as iPerturb
import torch
import scanpy as sc

cuda = torch.cuda.is_available()
if cuda:
    print('cuda is available')
else:
    print('cuda is not available')

# 导入必要的数据集，以及batch_key,condition_key,groundtruth_key(可省略)等参数
dataset = 'Pbmc'
batch_key = 'batch'
condition_key = 'condition'
groundtruth_key = 'groundtruth' ## 用于计算ARI

print(dataset+' done!')
# real data
anndata = sc.read_h5ad('/data/chenyz/iPerturb_project/data/' +dataset +'.h5ad')
savepath = '/data/chenyz/iPerturb_project/Score/result/Result/'+ dataset


datasets,raw,var_names,index_names = iPerturb.preprocess.data_load(anndata, batch_key = batch_key ,condition_key = condition_key ,groundtruth_key = groundtruth_key ,n_top_genes = 4000)
hyper = iPerturb.utils.create_hyper(datasets,var_names,index_names)
# 训练模型
epochs = 30

svi,scheduler,iPerturb_model = iPerturb.model.model_init_(hyper, latent_dim1=100, latent_dim2=30, latent_dim3=30, 
                                                            optimizer = Adam, lr = 0.006, gamma = 0.2, milestones = [20], 
                                                            set_seed=123, cuda = cuda, alpha = 1e-4)
x_pred,reconstruct_data = iPerturb.model.RUN(datasets,iPerturb_model,svi,scheduler,epochs,hyper,raw,cuda,batch_size=100,if_likelihood=True)

reconstruct_data.write(os.path.join(savepath, 'iPerturb.h5ad'))

```
    
`
