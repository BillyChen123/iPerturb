# iPerturb

iPerturb utilizes variational autoencoders for multi-condition integration of single-cell data at the population level. For more information, refer to the [PyPI page](https://pypi.org/project/iperturb/). Sample information can be found in the [vignettes](https://github.com/BillyChen123/iPerturb/blob/master/vignettes/vignettes.ipynb).

## Installation

Install using pip:

```bash
pip install iPerturb
```

## Usage

After installation, you can use functions and classes from the iPerturb package in your Python code.

For example:

```python
import iPerturb.iperturb as iPerturb
import torch
import scanpy as sc

cuda = torch.cuda.is_available()
if cuda:
    print('cuda is available')
else:
    print('cuda is not available')

# Load necessary datasets and parameters such as batch_key, condition_key, and groundtruth_key (optional)
dataset = 'Pbmc'
batch_key = 'batch'
condition_key = 'condition'
groundtruth_key = 'groundtruth'  # Used for calculating ARI

print(dataset + ' done!')

# Load real data
anndata = sc.read_h5ad('/data/chenyz/iPerturb_project/data/' + dataset + '.h5ad')
savepath = '/data/chenyz/iPerturb_project/Score/result/Result/' + dataset

datasets, raw, var_names, index_names = iPerturb.preprocess.data_load(anndata, batch_key=batch_key, condition_key=condition_key, groundtruth_key=groundtruth_key, n_top_genes=4000)
hyper = iPerturb.utils.create_hyper(datasets, var_names, index_names)

# Train the model
epochs = 30
optimizer = torch.optim.Adam

svi, scheduler, iPerturb_model = iPerturb.model.model_init_(hyper, latent_dim1=100, latent_dim2=30, latent_dim3=30, 
                                                            optimizer=optimizer, lr=0.006, gamma=0.2, milestones=[20], 
                                                            set_seed=123, cuda=cuda, alpha=1e-4)

x_pred, reconstruct_data = iPerturb.model.RUN(datasets, iPerturb_model, svi, scheduler, epochs, hyper, raw, cuda, batch_size=100, if_likelihood=True)

reconstruct_data.write(os.path.join(savepath, 'iPerturb.h5ad'))
```
    
`
