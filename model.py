# setup environment
import os

from . import preprocess


# various import statements
import torch
import torch.nn as nn
from torch.nn.functional import softplus, softmax
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape
from pyro.optim import MultiStepLR
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO, RenyiELBO
from torch.utils.data import DataLoader
import numpy as np

import time

from . import utils

####################### 构建编码解码器的函数 ###########################
# Helper for making fully-connected neural networks
def make_fc(dims):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])  # Exclude final ReLU non-linearity

# Helper for making fully-connected neural networks
def make_fc2(dims):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])  # Exclude final ReLU non-linearity

# Splits a tensor in half along the final dimension
def split_in_half(t):
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)

# Helper for broadcasting inputs to neural net
def broadcast_inputs(input_args):
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args

####################### 完成需要的编码-解码器 ###########################
# decoder:
# total_special_decoder(z_total,z_special)
# x_decoder(z)
# encode:
# z_encoder(x)
# total_special_encoder(z)

######################## Decoder Part #################################
# Used in parameterizing p(z |z_total,z_special)
class ZSDecoder(nn.Module):
    def __init__(self, num_batch, z_special_dim, z_total_dim, hidden_dims1, hidden_dims2):
        super().__init__()
        dims = [num_batch + z_total_dim] + hidden_dims1 + [2 * z_special_dim]
        self.fc = make_fc(dims)

    def forward(self, t ,z_total):
        z_t = broadcast_inputs([z_total, t])
        z_t = torch.cat(z_t, dim=-1)
        # We reshape the input to be two-dimensional so that nn.BatchNorm1d behaves correctly
        _z_t = z_t.reshape(-1, z_t.size(-1))
        hidden = self.fc(_z_t)
        # If the input was three-dimensional we now restore the original shape
        hidden = hidden.reshape(z_t.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = softplus(scale)
        return loc, scale


# Used in parameterizing p(x |z)
class XDecoder(nn.Module):
    # This __init__ statement is executed once upon construction of the neural network.
    # Here we specify that the neural network has input dimension z2_dim
    # and output dimension 2 * num_genes.
    def __init__(self, num_genes, z_dim, hidden_dims1, hidden_dims2):
        super().__init__()
        dims = [z_dim] + hidden_dims1 + [2 * num_genes]
        self.fc = make_fc(dims)


    def forward(self, z):
        loc, scale = split_in_half(self.fc(z))
        # Note that mu is normalized so that total count information is
        # encoded by the latent variable ℓ.
        scale = softplus(scale)
        return loc, scale


# Used in parameterizing q(z_total | z) and q(z_special | z)
class ZDecoder(nn.Module):
    def __init__(self, z_dim, z_total_dim, hidden_dims1, hidden_dims2):
        super().__init__()
        dims = [z_total_dim] + hidden_dims1  + [2 * z_dim]
        self.fc = make_fc2(dims)

    def forward(self, z_total):
        loc, scale = split_in_half(self.fc(z_total))
        # Note that mu is normalized so that total count information is
        # encoded by the latent variable ℓ.
        scale = softplus(scale)
        return loc, scale

# Used in parameterizing q(z_total | z) and q(z_special | z)
class Z2Decoder(nn.Module):
    def __init__(self, z_dim, z_total_dim, z_special_dim, hidden_dims1, hidden_dims2):
        super().__init__()
        dims = [z_total_dim + z_special_dim] + hidden_dims1  + [2 * z_dim]
        self.fc = make_fc2(dims)

    def forward(self, z_total, z_special):
        z_total_special = torch.cat([z_total,z_special], dim=-1)
        # We reshape the input to be two-dimensional so that nn.BatchNorm1d behaves correctly
        _z_total_special = z_total_special.reshape(-1, z_total_special.size(-1))
        hidden = self.fc(_z_total_special)
        # If the input was three-dimensional we now restore the original shape
        hidden = hidden.reshape(_z_total_special.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        # Here and elsewhere softplus ensures that scale is positive. Note that we generally
        # expect softplus to be more numerically stable than exp.
        scale = softplus(scale)
        return loc, scale


######################## Encoder Part #################################

class TSEncoder(nn.Module):
    def __init__(self, z_dim, z_special_dim, hidden_dims1, hidden_dims2):
        super().__init__()
        dims = [z_dim] + hidden_dims1 + [2 * z_special_dim]
        self.fc = make_fc(dims)
        self.z_special_dim = z_special_dim

    def forward(self, z):
        # Transform the counts x to log space for increased numerical stability.
        # Note that we only use this transformation here; in particular the observation
        # distribution in the model is a proper count distribution.
        loc, scale = split_in_half(self.fc(z))
        # Here and elsewhere softplus ensures that scale is positive. Note that we generally
        # expect softplus to be more numerically stable than exp.
        scale = softplus(scale)
        return loc, scale



class ZSEncoder(nn.Module):
    def __init__(self, num_batch, z_special_dim, hidden_dims1, hidden_dims2):
        super().__init__()
        dims = [z_special_dim] + hidden_dims1 + [num_batch]
        self.fc = make_fc(dims)

    def forward(self, z_special):
        # Transform the counts x to log space for increased numerical stability.
        # Note that we only use this transformation here; in particular the observation
        # distribution in the model is a proper count distribution.
        logits = self.fc(z_special)
        return logits

class ZS2Encoder(nn.Module):
    def __init__(self, num_batch, z_special_dim, z_total_dim, hidden_dims1, hidden_dims2):
        super().__init__()
        dims = [z_special_dim + num_batch] + hidden_dims1 + [2 * z_total_dim]
        self.fc = make_fc(dims)
        self.z_special_dim = z_special_dim

    def forward(self, z_special, t):
        # This broadcasting is necessary since Pyro expands y during enumeration (but not z2)
        zs_t = broadcast_inputs([z_special, t])
        zs_t = torch.cat(zs_t, dim=-1)
        # We reshape the input to be two-dimensional so that nn.BatchNorm1d behaves correctly
        _zs_t = zs_t.reshape(-1, zs_t.size(-1))
        hidden = self.fc(_zs_t)
        # If the input was three-dimensional we now restore the original shape
        hidden = hidden.reshape(zs_t.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = softplus(scale)
        return loc, scale

# Used in parameterizing q(z_total,t | x)
class ZEncoder(nn.Module):
    def __init__(self, z_dim, num_genes, hidden_dims1, hidden_dims2):
        super().__init__()
        dims = [num_genes] + hidden_dims1 + [2 * z_dim]
        self.fc = make_fc(dims)

    def forward(self, x):
        # Transform the counts x to log space for increased numerical stability.
        # Note that we only use this transformation here; in particular the observation
        # distribution in the model is a proper count distribution.
        h1, h2 = split_in_half(self.fc(x))
        loc, scale = h1, softplus(h2)
        return loc, scale

####################### 构建pyro模型 ###########################
class iPerturb_model_init(nn.Module):
    def __init__(self, num_genes, num_batch, latent_dim1=20, latent_dim2=10, latent_dim3=10, alpha=1e-5, scale_factor=1.0):
        self.num_genes = num_genes
        self.num_batch = num_batch

        # This is the latent_dim dimension of both z, z_total, z_special
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        self.latent_dim3 = latent_dim3

        # This hyperparameter controls the strength of the auxiliary classification loss
        self.alpha = alpha
        self.scale_factor = scale_factor

        super().__init__()

        # Setup the various neural networks used in the model and guide
        self.z_decoder = ZDecoder(z_dim=self.latent_dim1, z_total_dim=self.latent_dim2,
                                    hidden_dims1=[100], hidden_dims2=[50])

        self.z2_decoder = Z2Decoder(z_dim=self.latent_dim1, z_total_dim=self.latent_dim2,
                                    z_special_dim=self.latent_dim3, hidden_dims1=[100], hidden_dims2=[50])

        self.zs_decoder = ZSDecoder(num_batch=self.num_batch ,z_special_dim=self.latent_dim3, z_total_dim=self.latent_dim2, hidden_dims1=[100], hidden_dims2=[50])

        self.x_decoder = XDecoder(num_genes=num_genes, hidden_dims1=[100], hidden_dims2=[50], z_dim=self.latent_dim1)


        self.ts_encoder = TSEncoder(z_dim=self.latent_dim1,
                                    z_special_dim=self.latent_dim3 ,hidden_dims1=[100], hidden_dims2=[50])

        self.zs_encoder = ZSEncoder(num_batch=self.num_batch, z_special_dim=self.latent_dim3 ,hidden_dims1=[100], hidden_dims2=[50])

        self.z_encoder = ZEncoder(z_dim=self.latent_dim1, num_genes=num_genes, hidden_dims1=[100], hidden_dims2=[50])

        self.zs2_encoder = ZS2Encoder(  num_batch=self.num_batch, 
                                        z_total_dim=self.latent_dim2, z_special_dim=self.latent_dim3 ,hidden_dims1=[100], hidden_dims2=[50])

        self.epsilon = 0.006

    def model(self, x, one_hot):
        # Register various nn.Modules (i.e. the decoder/encoder networks) with Pyro
        pyro.module("iPerutrb_model", self)

        # This gene-level parameter modulates the variance of the observation distribution
        theta = pyro.param("inverse_dispersion", 10.0 * x.new_ones(self.num_genes),
                        constraint=constraints.positive)

        # We scale all sample statements by scale_factor so that the ELBO loss function
        # is normalized wrt the number of datapoints and genes.
        # This helps with numerical stability during optimization.
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z_total = pyro.sample("z_total", dist.Normal(0, x.new_ones(self.latent_dim2)).to_event(1))
            t_logit = pyro.sample("t_logit", dist.Dirichlet(x.new_ones(self.num_batch)))
            t = pyro.sample("t", dist.OneHotCategorical(logits=t_logit), obs=one_hot)

            z_special_loc, z_special_scale = self.zs_decoder(z_total, t_logit) # t_logit or t_smooth
            z_special = pyro.sample("z_special", dist.Normal(z_special_loc, z_special_scale).to_event(1))

            z_loc, z_scale = self.z2_decoder(z_total, z_special)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            x_loc, x_scale = self.x_decoder(z)
            x_scale = softplus(x_scale + theta + self.epsilon)

            x = pyro.sample("x", dist.Normal(x_loc, x_scale).to_event(1), obs=x)


    # The guide specifies the variational distribution
    def guide(self, x, one_hot):
        pyro.module("iPerutrb_model", self)

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z_loc, z_scale = self.z_encoder(x)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            z_special_loc, z_special_scale = self.ts_encoder(z)
            z_special = pyro.sample("z_special", dist.Normal(z_special_loc, z_special_scale).to_event(1))

            t_logit = self.zs_encoder(z_special)

            t_dist = dist.OneHotCategorical(logits=t_logit)
            if one_hot is None:
                t = pyro.sample("t", t_dist)
            else:
                # x is labeled so add a classification loss term
                # (this way q(y|z2) learns from both labeled and unlabeled data)
                classification_loss = t_dist.log_prob(one_hot)
                # Note that the negative sign appears because we're adding this term in the guide
                # and the guide log_prob appears in the ELBO as -log q
                pyro.factor("classification_loss", -self.alpha * classification_loss, has_rsample=False)

            z_total_loc, z_total_scale = self.zs2_encoder(z_special, t_logit)
            z_total = pyro.sample("z_total", dist.Normal(z_total_loc, z_total_scale).to_event(1))

    
# 用来记录pyro模型迭代的重要参数，并进行模型初始化
def model_init_(hyper, latent_dim1, latent_dim2, latent_dim3, optimizer, lr, gamma, milestones, set_seed=123, cuda = False, alpha=0):
    num_genes = hyper['num_genes']
    num_batch = hyper['num_batch']
    scale_factor = 1.0 / (hyper['batch_size'] * hyper['num_genes'])
    # Clear Pyro param store so we don't conflict with previous
    # # training runs in this session
    # pyro.clear_param_store()
    # Fix random number seed
    pyro.util.set_rng_seed(set_seed)
    # Enable optional validation warnings
    pyro.enable_validation(True)
    # Instantiate instance of model/guide and various neural networks
    iPerturb_model = iPerturb_model_init(num_genes=num_genes,num_batch=num_batch,latent_dim1=latent_dim1, latent_dim2=latent_dim2, latent_dim3=latent_dim3, scale_factor=scale_factor,alpha=alpha)
    if cuda == True:
        iPerturb_model = iPerturb_model.cuda()
    # Setup an optimizer (Adam) and learning rate scheduler.
    # We start with a moderately high learning rate (0.006) and
    # reduce by a factor of 5 after 20 epochs.
    scheduler = MultiStepLR({'optimizer': optimizer,
                            'optim_args': {'lr': lr},
                            'gamma': gamma, 'milestones': milestones})

    # Tell Pyro to enumerate out y when y is unobserved.
    # (By default y would be sampled from the guide)
    guide = config_enumerate(iPerturb_model.guide, "parallel", expand=True)
    # Setup a variational objective for gradient-based learning.
    # Note we use TraceEnum_ELBO in order to leverage Pyro's machinery
    # for automatic enumeration of the discrete latent variable y.


    elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    # elbo = RenyiELBO(strict_enumeration_warning=False)
    
    svi = SVI(iPerturb_model.model, guide, scheduler, elbo)
    return svi, scheduler, iPerturb_model

## 利用pyro进行模型迭代
def RUN_model(epochs,svi,scheduler,dataloader):
    pyro.clear_param_store()
    start_time = time.time()
    for epoch in range(epochs):
        losses = []

        # Take a gradient step for each mini-batch in the dataset
        for x,one_hot,_ in dataloader:
            if one_hot is not None:
                one_hot = one_hot.type_as(x)

            loss = svi.step(x,one_hot)
            losses.append(loss)

        # Tell the scheduler we've done one epoch.
        scheduler.step()

        print("[Epoch %02d]  Loss: %.5f" % (epoch, np.mean(losses)))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Finished training! Time:{execution_time}.")

# 总的RUN
def RUN(datasets,iPerturb_model,svi,scheduler,epochs,hyper,raw,cuda,batch_size=100,if_likelihood=False):
    sc_datasets = preprocess.scDataset(datasets,cuda=cuda,transform=preprocess.transform_numpy)
    batch_sampler = preprocess.BatchSampler(datasets,batch_size=batch_size,cuda=cuda)

    dataloader = DataLoader(sc_datasets, batch_sampler=batch_sampler)
    dataloader2 = DataLoader(sc_datasets, batch_size=batch_size, shuffle=True)

    # training runs in this session
    RUN_model(epochs,svi,scheduler,dataloader)
    # Put the neural networks in evaluation mode (needed because of batch norm)
    iPerturb_model.eval()
    if cuda:
        # z
        z_pred = iPerturb_model.z_encoder(hyper['x'].cuda())[0]
    else:
        # z
        z_pred = iPerturb_model.z_encoder(hyper['x'])[0]
    # 重构
    x_pred = iPerturb_model.x_decoder(z_pred)[0]
    z_s = iPerturb_model.ts_encoder(z_pred)[0]
    t_logit = iPerturb_model.zs_encoder(z_s)
    z_t = iPerturb_model.zs2_encoder(z_s, t_logit)[0]


    x_pred = x_pred.data.cpu().numpy()
    z_pred = z_pred.data.cpu().numpy()
    zs_pred = z_s.data.cpu().numpy()
    zt_pred = z_t.data.cpu().numpy()
     

    # likelihood
    if if_likelihood:
        RUN_model(epochs,svi,scheduler,dataloader2)
        iPerturb_model.eval()
        if cuda:
            # z
            z_pred_ = iPerturb_model.z_encoder(hyper['x'].cuda())[0]
        else:
            # z
            z_pred_ = iPerturb_model.z_encoder(hyper['x'])[0]
        zs_pred_ = iPerturb_model.ts_encoder(z_pred_)[0]
        t_logits = iPerturb_model.zs_encoder(zs_pred_)
        t_logits = softmax(t_logits,dim=-1).data.cpu().numpy()
        likelihood = np.argmax(t_logits, axis=-1)


    # 创建 AnnData 对象
    datasets = utils.construct_anndata(data = x_pred, columns_names = hyper['var_names'], index_names = hyper['index_names'], raw=raw)  
        
    datasets.obsm['x_emb'] = z_pred
    datasets.obsm['zt_emb'] = zt_pred
    datasets.obsm['zs_emb'] = zs_pred

    if if_likelihood:
        datasets.obsm['t_logits'] = t_logits
        datasets.obs['likelihood'] = likelihood
        datasets.obs['t_logits'] = datasets.obsm['t_logits'][:,0]

    # ari_max,index_i = utils.Find_ari(datasets)
    # print(f'第一层重构已完成,它的ari是{ari_max},它的resolution是{index_i}.')


    return x_pred,datasets
